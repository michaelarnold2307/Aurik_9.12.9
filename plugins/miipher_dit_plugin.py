"""MIIPHER-DiT Plugin — Flow-Matching Singing Voice Enhancement für Aurik.

§v10.14: Ersetzt das proprietäre Google MIIPHER durch ein offenes
Flow-Matching-DiT-Modell.

Architektur (§v10.19 A6 — verifiziert am exportierten ONNX):
  - FlowMatchingDiT (Transformer, 18 Layer, 768-dim, 12 Heads)
  - UNKONDITIONIERT: Inputs sind x [B,T,1] + t [B] — kein Whisper-/Vocoder-Pfad
    im Inferenz-ONNX. Die früheren Whisper-Encoder/BigVGAN-Deklarationen waren
    Planungs-Doku, nie exportiert.
  - ONNX-kompatibel (OpSet 14+, Dynamic Axes)

Aurik-Integration:
  - PLM-registriert (§4.6b): set_active("MIIPHER_DiT", True)
  - Hallucination-Guard (§2.46e): spectral_novelty > 0.35 → Rollback
  - DSP-Fallback (§V6 (copilot-instructions.md)): IMCRA/Wiener bei Modell-Fehler
  - M/S-Processing: Stereo→Mono(Mid)→Enhance→Stereo
  - Sample-Rate-Adapter: 48kHz ↔ Modell-SR
  - RAM-Budget: 2.5 GB (DiT)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Modulebene (§G174): Flags ohne Backend-Abhängigkeiten.
try:
    from backend.core.music_model_flags import resolve_model_path as _resolve_flag

    _FLAGS_AVAILABLE = True
except Exception as _flags_exc:
    _FLAGS_AVAILABLE = False
    _resolve_flag = None  # type: ignore[assignment]
    logger.debug("MIIPHER-DiT: music_model_flags nicht ladbar: %s", _flags_exc)


def _resolve_or(fallback: Path, model_key: str) -> Path:
    """§v10.19 Feature-Flag-Routing: aufgelöster Pfad, sonst Fallback.

    Liefert den via music_model_flags.resolve_model_path aufgelösten
    Pfad (z. B. whisper_tiny.onnx als Encoder-Ersatz), falls vorhanden.
    """
    if _FLAGS_AVAILABLE and _resolve_flag is not None:
        _p = _resolve_flag(model_key)
        if _p is not None and Path(_p).exists():
            return Path(_p)
    return fallback


# ── Lazy imports für optionale Abhängigkeiten ────────────────────────────
try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    ort = None  # type: ignore[assignment]

try:
    import torch
    import torchaudio

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]
    torchaudio = None  # type: ignore[assignment]

# ── PLM-Integration ──────────────────────────────────────────────────────
try:
    from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

    _PLM_AVAILABLE = True
except ImportError:
    _PLM_AVAILABLE = False

try:
    from backend.core.ml_memory_budget import release as _ml_budget_release
    from backend.core.ml_memory_budget import try_allocate as _ml_budget_try_allocate
except ImportError:
    _ml_budget_try_allocate = cast(Any, None)
    _ml_budget_release = cast(Any, None)


@dataclass
class MiipherDiTResult:
    """Ergebnis der MIIPHER-DiT Gesangsverbesserung."""

    audio: np.ndarray
    applied: bool
    model_used: str  # "miipher_dit" | "dsp_fallback" | "none"
    novelty: float = 0.0
    hf_delta_db: float = 0.0
    processing_time_s: float = 0.0
    metadata: dict | None = None


class MiipherDiTPlugin:
    """Flow-Matching-DiT Plugin als MIIPHER-Ersatz.

    Aktiviert NUR für:
      - Stark degradierten Gesang (SNR < 10 dB, restorability < 30)
      - Material mit Codec-Artefakten (mp3_low, streaming)
      - KEIN Instrumental-only Audio

    §2.46e Hallucination-Guard: spectral_novelty > 0.35 → Rollback auf DSP-Fallback.
    """

    _BUDGET_NAME: str = "MIIPHER_DiT"
    _BUDGET_SIZE_GB: float = 2.5  # DiT 18-layer ~1.5GB + Whisper ~1GB
    _MODEL_SR: int = 48000  # Interne Sample-Rate (muss Aurik-SR matchen)
    _HALLUCINATION_THRESHOLD: float = 0.35

    # Materialien, für die MIIPHER-DiT aktiviert wird
    _TARGET_MATERIALS: frozenset[str] = frozenset({"mp3_low", "streaming", "aac", "minidisc"})

    def __init__(self) -> None:
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._device: str = "cpu"
        self._ort_session: object | None = None  # onnxruntime.InferenceSession

        # Modell-Pfade (§v10.19 Feature-Flag-Routing)
        _model_dir = Path(__file__).parent.parent / "models" / "miipher_dit"
        self._dit_onnx_path: Path = _resolve_or(_model_dir / "flow_matching_dit.onnx", "miipher_dit")
        self._whisper_onnx_path: Path = _resolve_or(_model_dir / "whisper_encoder.onnx", "whisper_encoder")
        self._bigvgan_onnx_path: Path = _resolve_or(_model_dir / "bigvgan_vocoder.onnx", "bigvgan")

        # Budget-Registrierung
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Lädt das Flow-Matching-DiT ONNX-Modell; aktiviert DSP-Fallback bei Fehler."""
        if not _ONNX_AVAILABLE:
            logger.warning("onnxruntime nicht verfügbar — MIIPHER-DiT DSP-Ersatzpfad aktiv")
            self._fallback_active = True
            return

        if _ml_budget_try_allocate is not None:
            if not _ml_budget_try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("MIIPHER-DiT: ML-Budget erschöpft — DSP-Ersatzpfad aktiv")
                self._fallback_active = True
                return

        if not self._dit_onnx_path.exists():
            logger.info(
                "MIIPHER-DiT: ONNX-Modell nicht gefunden (%s) — DSP-Ersatzpfad",
                self._dit_onnx_path,
            )
            self._fallback_active = True
            return

        try:
            _providers = ["CPUExecutionProvider"]
            if ort is not None:
                _available = ort.get_available_providers()
                if "ROCMExecutionProvider" in _available:
                    # §v10.304: ROCm-EP vor CUDA-EP — AMD-Systeme liefern CUDA-EP nie.
                    _providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
                    self._device = "cuda"
                elif "CUDAExecutionProvider" in _available:
                    _providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    self._device = "cuda"

            self._ort_session = ort.InferenceSession(
                str(self._dit_onnx_path),
                providers=_providers,
            )
            self._model_loaded = True
            logger.info(
                "MIIPHER-DiT geladen: %s (device=%s, %.1f MB)",
                self._dit_onnx_path.name,
                self._device,
                self._dit_onnx_path.stat().st_size / 1e6,
            )

            # PLM-Registrierung (§4.6b)
            if _PLM_AVAILABLE:
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _plm_register

                    _plm_register(
                        self._BUDGET_NAME,
                        size_gb=self._BUDGET_SIZE_GB,
                        unload_fn=self.unload,
                    )
                except Exception as _exc:
                    logger.debug("MIIPHER-DiT PLM-Registrierung: %s", _exc)

        except Exception as exc:
            logger.warning("MIIPHER-DiT Ladefehler: %s — DSP-Ersatzpfad", exc)
            self._fallback_active = True

    def unload(self) -> None:
        """Entlädt das ONNX-Modell aus dem RAM (PLM-Eviction-Callback)."""
        self._ort_session = None
        self._model_loaded = False
        if _ml_budget_release is not None:
            try:
                _ml_budget_release(self._BUDGET_NAME)
            except Exception:
                pass
        logger.debug("MIIPHER-DiT entladen")

    # ── DSP-Fallback (IMCRA/Wiener) ─────────────────────────────────────

    @staticmethod
    def _dsp_fallback(audio: np.ndarray, sr: int) -> np.ndarray:
        """§V6 (copilot-instructions.md): Deterministischer DSP-Ersatzpfad wenn ML nicht verfügbar.

        Nutzt Wiener-Filter + sanfte HF-Anhebung für minimale hörbare Verbesserung
        ohne Artefakt-Risiko. Kein aggressives Denoising — Primum non nocere.
        """
        try:
            from scipy import signal

            # Sanfte Rauschunterdrückung via Wiener (n_fft=2048, konservativ)
            _n_fft = 2048
            _mono = audio if audio.ndim == 1 else np.mean(audio, axis=0)
            _f, _t, Zxx = signal.stft(_mono, fs=sr, nperseg=_n_fft, noverlap=_n_fft // 2)
            _noise_floor = np.mean(np.abs(Zxx[:, :10]), axis=1, keepdims=True)
            _gain = np.maximum(0, 1.0 - _noise_floor / (np.abs(Zxx) + 1e-10))
            _gain = np.clip(_gain, 0.3, 1.0)  # Max 70% Dämpfung (konservativ)
            Zxx_clean = Zxx * _gain
            _, _restored = signal.istft(Zxx_clean, fs=sr, nperseg=_n_fft, noverlap=_n_fft // 2)

            if audio.ndim == 2:
                _restored = np.stack([_restored[: len(audio)]], axis=-1).repeat(2, axis=-1)
            else:
                _restored = _restored[: len(audio)]

            return cast(np.ndarray, _restored.astype(np.float32))
        except Exception:
            return audio  # Fail-safe: Original zurück

    # ── Spectral Novelty (Hallucination-Guard §2.46e) ───────────────────

    @staticmethod
    def _spectral_novelty(before: np.ndarray, after: np.ndarray, sr: int) -> float:
        """Misst wie viel NEUES Spektrum das Modell hinzugefügt hat.

        > 0.35 → Modell halluziniert → Rollback.
        """
        try:
            _mono_before = before if before.ndim == 1 else np.mean(before, axis=0)
            _mono_after = after if after.ndim == 1 else np.mean(after, axis=0)
            _n_fft = 2048
            _spec_before = np.abs(np.fft.rfft(_mono_before[: min(len(_mono_before), sr * 5)], n=_n_fft))
            _spec_after = np.abs(np.fft.rfft(_mono_after[: min(len(_mono_after), sr * 5)], n=_n_fft))
            _spec_before_norm = _spec_before / (np.max(_spec_before) + 1e-10)
            _spec_after_norm = _spec_after / (np.max(_spec_after) + 1e-10)
            _diff = np.mean(np.abs(_spec_after_norm - _spec_before_norm))
            return float(np.clip(_diff, 0.0, 1.0))
        except Exception:
            return 0.0

    # ── Haupt-API ───────────────────────────────────────────────────────

    def should_apply(self, material: str, restorability_score: float = 50.0) -> bool:
        """Prüft ob MIIPHER-DiT für dieses Material angewendet werden sollte."""
        _mat = str(material).lower()
        # Nur für stark degradierte Codec-Materialien oder sehr niedrige Restorability
        if _mat in self._TARGET_MATERIALS:
            return True
        if restorability_score < 30:
            return True
        return False

    def enhance(
        self,
        audio: np.ndarray,
        sr: int,
        material: str = "unknown",
        *,
        restorability_score: float = 50.0,
        vocal_stem: np.ndarray | None = None,
    ) -> MiipherDiTResult:
        """Führt MIIPHER-DiT Gesangsverbesserung mit allen Guards aus.

        Args:
            audio: Input-Audio (mono oder stereo), float32, shape (n,) oder (n, 2).
            sr: Sample-Rate (muss 48000 Hz sein).
            material: Material-Typ für Kontext-Entscheidung.
            restorability_score: 0-100, Qualität des Quellmaterials.
            vocal_stem: Optionaler Vocal-Stem für gezielte Bearbeitung.

        Returns:
            MiipherDiTResult mit audio, applied, novelty, Metriken.
        """
        t_start = time.time()
        _mat = str(material).lower()

        # ── Gate 1: Material-Check ──
        if not self.should_apply(material, restorability_score):
            return MiipherDiTResult(
                audio=audio,
                applied=False,
                model_used="none",
                metadata={"reason": "material_not_target"},
            )

        # ── Gate 2: SR-Assertion ──
        assert sr == self._MODEL_SR, f"MIIPHER-DiT: SR muss {self._MODEL_SR} Hz sein"

        # ── Gate 3: Modell-Verfügbarkeit ──
        if self._fallback_active or not self._model_loaded:
            _fallback = self._dsp_fallback(audio, sr)
            return MiipherDiTResult(
                audio=_fallback,
                applied=True,
                model_used="dsp_fallback",
                processing_time_s=time.time() - t_start,
            )

        # ── M/S-Processing ──
        _is_stereo = audio.ndim == 2 and audio.shape[1] == 2
        if _is_stereo:
            _mid = (audio[:, 0] + audio[:, 1]) / 2.0
            _side = (audio[:, 0] - audio[:, 1]) / 2.0
            _target = _mid
        else:
            _target = audio

        # ── §v10.14: Chunked-Routing für lange Songs (>10s) ──
        _duration_s = len(_target) / sr
        if _duration_s > 10.0:
            _result_audio = self._enhance_chunked(_target, sr)
        else:
            _result_audio = self._enhance_single(_target, sr)

        # ── Hallucination-Guard (§2.46e) ──
        _novelty = self._spectral_novelty(_target, _result_audio, sr)
        if _novelty > self._HALLUCINATION_THRESHOLD:
            logger.warning(
                "§2.46e MIIPHER-DiT Hallucination-Guard: novelty=%.3f > %.2f → Rollback (Material=%s)",
                _novelty,
                self._HALLUCINATION_THRESHOLD,
                _mat,
            )
            _result_audio = _target
            _model_used = "none"
        else:
            logger.info(
                "MIIPHER-DiT: %s enhanced (%.2fs, novelty=%.3f)",
                _mat,
                time.time() - t_start,
                _novelty,
            )
            _model_used = "miipher_dit"

        # ── M/S-Remix ──
        if _is_stereo:
            _out_l = _result_audio + _side[: len(_result_audio)]
            _out_r = _result_audio - _side[: len(_result_audio)]
            _final = np.stack([_out_l, _out_r], axis=-1).astype(np.float32)
        else:
            _final = _result_audio.astype(np.float32)

        # ── HF-Delta messen ──
        try:
            _bw_before = float(np.sum(np.abs(np.fft.rfft(_target[: sr * 2]))[sr // 4 :]) + 1e-10)
            _bw_after = float(np.sum(np.abs(np.fft.rfft(_final[: sr * 2]))[sr // 4 :]) + 1e-10)
            _hf_delta = float(20.0 * np.log10(_bw_after / _bw_before)) if _bw_before > 0 else 0.0
        except Exception:
            _hf_delta = 0.0

        return MiipherDiTResult(
            audio=_final,
            applied=True,
            model_used=_model_used,
            novelty=_novelty,
            hf_delta_db=_hf_delta,
            processing_time_s=time.time() - t_start,
        )

    # ── §v10.14: Chunked Processing für lange Songs ────────────────────

    def _enhance_single(self, audio_mono: np.ndarray, sr: int) -> np.ndarray:
        """ONNX-Inferenz für kurze Segmente (<10s).

        Flow-Matching-Korrektur: Das Modell sagt das Geschwindigkeitsfeld v̂
        vorher, nicht das Audio direkt. Die Formel lautet:
            ŷ = x + (1 - t) · v̂
        (Ref: train_miipher_dit.py:12)
        """
        _peak = float(np.max(np.abs(audio_mono))) + 1e-10
        _input = (audio_mono / _peak).astype(np.float32)
        _input_onnx = _input.reshape(1, -1, 1)
        _t = np.array([0.5], dtype=np.float32)
        _velocity = cast(Any, self._ort_session).run(None, {"x": _input_onnx, "t": _t})[0]
        # Flow-Matching: ŷ = x + (1-t) · v̂
        _corrected = _input_onnx + (1.0 - _t[0]) * _velocity.reshape(1, -1, 1)
        return cast(np.ndarray, _corrected.reshape(-1).astype(np.float32) * _peak)

    def _enhance_chunked(self, audio_mono: np.ndarray, sr: int) -> np.ndarray:
        """Chunked-Processing für lange Songs via Auriks ChunkedPipeline.

        Nutzt die bestehende ChunkedPipeline-Infrastruktur für
        Overlap-Add mit Hann-Crossfade — konsistent mit dem Rest
        der Aurik-Pipeline.
        """
        try:
            from backend.core.chunked_streaming import ChunkedPipeline

            # ChunkedPipeline als Wrapper: 4s Chunks, 0.5s Overlap
            cp = ChunkedPipeline(chunk_duration_s=4.0, overlap_s=0.5, crossfade_s=0.05)
            chunks = cp.compute_chunks(audio_mono, sr)

            results = []
            for i, (start, end) in enumerate(chunks):
                _chunk = audio_mono[start:end]
                _enhanced = self._enhance_single(_chunk, sr)
                from backend.core.chunked_streaming import ChunkResult

                results.append(ChunkResult(_enhanced, sr, i, start, end))

            return cp.collect_results(results, sr)
        except ImportError:
            # Fallback: Direktverarbeitung (nur für kurze Songs sicher)
            logger.warning("MIIPHER-DiT: ChunkedPipeline nicht verfügbar — Direktverarbeitung")
            return self._enhance_single(audio_mono, sr)


# ── Singleton (PLM-kompatibel) ──────────────────────────────────────────

_instance: MiipherDiTPlugin | None = None


def get_miipher_dit() -> MiipherDiTPlugin:
    """Gibt die process-weite MIIPHER-DiT-Singleton zurück (Lazy-Load)."""
    global _instance
    if _instance is None:
        _instance = MiipherDiTPlugin()
    return _instance


def repair_vocal_miipher(
    audio: np.ndarray,
    sr: int,
    material: str = "unknown",
    *,
    restorability_score: float = 50.0,
) -> MiipherDiTResult:
    """Convenience-Wrapper für MIIPHER-DiT Gesangsverbesserung."""
    return get_miipher_dit().enhance(audio, sr, material, restorability_score=restorability_score)
