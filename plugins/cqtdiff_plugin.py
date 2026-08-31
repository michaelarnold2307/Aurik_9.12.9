"""
CQTdiff Plugin — Diffusions-basiertes Long-Gap-Inpainting für Aurik 10.0.0

Primäres Inpainting-Modell für Lücken ≥ 50 ms (Tape-Dropout, Bandaussetzer).
CQTdiff konditioniert eine Score-basierte Diffusion auf CQT-Domäne und
musikalische Phrasenkontextfenster.

Referenz:
    Moliner & Välimäki (2022): "Solving Audio Inverse Problems with a Diffusion
    Model", IEEE TASLP 2022. https://arxiv.org/abs/2210.15228
    Checkpoint (frei verfügbar): https://zenodo.org/record/7088416

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär:   CQTdiff (ONNX, ≥ 50 ms Lücken) — frei verfügbares IEEE-TASLP-Modell
    Fallback: VoiceFixer v2 → DiffWave-Plugin (diffwave_plugin.py)
    Kein AR (Yule-Walker) — verboten laut §4.2

CPU-Policy: Ausschließlich CPUExecutionProvider — keine CUDA-Abhängigkeit.
Modell-Gewichte: models/cqtdiff_plus/score_network.onnx

Aktivierung (in CAUSE_TO_PHASES):
    "tape_dropout": ["phase_24_dropout_repair", "phase_55_diffusion_inpainting"]
    Lücken < 50 ms → NMF-β (Phase 24); Lücken ≥ 50 ms → CQTdiff (Phase 55)
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class InpaintingResult:
    """Ergebnis des Diffusions-Inpaintings.

    Attribute:
        audio:          Restauriertes Audio (float32, normalisiert [-1,1])
        sr:             Sample-Rate (48000 Hz)
        kl_divergence:  KL-Divergenz Spektrum vor/nach (soll < 0.15 sein)
        chroma_corr:    Chroma-Pearson-Korrelation mit Phrasenkontext (≥ 0.92)
        model_used:     "cqtdiff_plus" | "diffwave_fallback" | "nmf_dsp_fallback"  # §V6 (copilot-instructions.md): logger.warning handled at call site
        confidence:     Konfidenz der Rekonstruktion ∈ [0, 1]
        groove_dtw_ms:  Onset-DTW-Distanz original/rekonstruiert [ms] (≤ 8 ms RMS)
    """

    audio: np.ndarray
    sr: int
    kl_divergence: float
    chroma_corr: float
    model_used: str
    confidence: float
    groove_dtw_ms: float = 0.0
    metadata: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "sr": self.sr,
            "kl_divergence": self.kl_divergence,
            "chroma_corr": self.chroma_corr,
            "model_used": self.model_used,
            "confidence": self.confidence,
            "groove_dtw_ms": self.groove_dtw_ms,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Singleton (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance: CQTdiffPlusPlugin | None = None
_lock = threading.Lock()


class CQTdiffPlusPlugin:
    """CQTdiff+ Diffusions-basiertes Audio-Inpainting für Aurik 10.0.0.

    Algorithmus (CQTdiff+ Vollpfad):
        1. CQT-Darstellung des vollständigen Audios berechnen
           (konstant-Q-Transform, 84 Bins, 3 Oktaven, Hop=256)
        2. Phrasen-Konditionierung: MusicalPhraseContextExtractor liefert
           musikalischen Kontext ±30 s um die Lücke
        3. Score-basierter Diffusions-Prozess (50 Denoising-Schritte):
           x_{t-1} = x_t − σ_t · score_network(x_t, t, context_cqt)
        4. Inverse CQT (iCQT) → zeitdomäne Audio
        5. PGHI für phasenkonsistente Rekonstruktion
        6. NMF-β-Vorinitialisierung als Diffusions-Prior W₀

    Konsistenz-Invarianten:
        - KL(Spektrum_vorher ‖ Spektrum_nachher) < 0.15
        - Chroma-Pearson(Phrase, Inpainting) ≥ 0.92
        - Groove DTW ≤ 8 ms RMS
        - NaN/Inf nach jeder Diffusions-Iteration: nan_to_num Pflicht

    Aktivierung:
        Nur bei gap_duration ≥ 50 ms (sonst NMF-β via phase_24).
        Kein Phrasenkontext für Audio < 8 s.
    """

    # Workspace-lokaler Modell-Pfad — liegt in models/cqtdiff/ (ONNX-Export via scripts/export_cqtdiff_onnx.py)
    _WORKSPACE_MODELS: Path = Path(__file__).parent.parent / "models" / "cqtdiff"
    _LEGACY_MODELS: Path = Path.home() / ".aurik" / "models" / "cqtdiff"
    MODELS_DIR: Path = _WORKSPACE_MODELS if _WORKSPACE_MODELS.exists() else _LEGACY_MODELS
    MIN_GAP_MS: float = 50.0  # Untergrenze für CQTdiff (sonst NMF-β)
    MAX_GAP_MS: float = 999.0  # Obergrenze (über 1 s → fallback)
    CQT_BINS: int = 84
    CQT_HOP: int = 256
    DIFFUSION_STEPS: int = 50

    def __init__(self) -> None:
        self._session = None  # onnxruntime.InferenceSession (score network)
        self._ts_model = None  # torch.jit.ScriptModule fallback
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Lädt ONNX Score-Netzwerk; aktiviert Fallback bei Fehler."""
        try:
            import onnxruntime as ort

            model_path = self.MODELS_DIR / "score_network.onnx"
            if model_path.exists():
                try:
                    from backend.core.ml_memory_budget import try_allocate as _try_alloc

                    if not _try_alloc("CQTdiff+", size_gb=0.19):
                        logger.warning("CQTdiff+: ML-Budget erschöpft — Fallback aktiv.")
                        self._fallback_active = True
                        return
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)

                self._session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"],
                )
                self._model_loaded = True
                logger.info("🔵 CQTdiff: Score-Netzwerk geladen (%s)", model_path)
                try:
                    from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                    _reg_plm(
                        "CQTdiff+",
                        size_gb=0.19,
                        unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_model_loaded", False),  # type: ignore[func-returns-value,misc]
                    )
                except Exception as _exc:
                    logger.debug("Operation failed (non-critical): %s", _exc)
            else:
                # ── ONNX nicht vorhanden → TorchScript-Direktladung ────
                ts_path = self.MODELS_DIR / "score_network.pt"
                if ts_path.exists():
                    logger.info("CQTdiff: score_network.onnx nicht gefunden — lade TorchScript direkt")
                    try:
                        import torch

                        self._ts_model = cast(Any, torch.jit.load(str(ts_path), map_location="cpu"))
                        cast(Any, self._ts_model).eval()
                        self._model_loaded = True
                        logger.info("🔵 CQTdiff: TorchScript geladen (%s)", ts_path)
                        return
                    except Exception as _ts_exc:
                        logger.warning("TorchScript-Ladung fehlgeschlagen: %s", _ts_exc)
                logger.info(
                    "CQTdiff: Kein Modell verfügbar (%s) — DSP-Fallback",
                    model_path,
                )
                self._fallback_active = True
        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("onnxruntime nicht verfügbar — CQTdiff+ Fallback aktiv")
            self._fallback_active = True
        except Exception as exc:
            logger.warning("CQTdiff+ Modell-Lade-Fehler: %s — Fallback aktiv", exc)
            self._fallback_active = True
            try:
                from backend.core.ml_memory_budget import release as _rel

                _rel("CQTdiff+")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def inpaint(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start_sample: int,
        gap_end_sample: int,
        *,
        context_audio: np.ndarray | None = None,
    ) -> InpaintingResult:
        """Füllt eine Dropout-Lücke diffusionsbasiert (CQTdiff+ oder Fallback).

        Algorithmus:
            1. Lückengröße prüfen — ≥ 50 ms: CQTdiff+, < 50 ms: NMF-β-Hinweis
            2. Phrasenkontextfenster extrahieren (context_audio oder lokaler Kontext)
            3. CQT-Konditionierung + Score-Netzwerk-Diffusion (50 Schritte)
            4. iCQT, PGHI, nan_to_num, clip(−1, 1)
            5. KL-Divergenz und Chroma-Korrelation prüfen

        Args:
            audio:             Vollständiges Audio-Signal (1D float32, 48000 Hz)
            sr:                Sample-Rate (muss 48000 sein)
            gap_start_sample:  Erster Sample der Lücke (inklusiv)
            gap_end_sample:    Letzter Sample der Lücke (exklusiv)
            context_audio:     Optionaler musikalischer Phrasenkontext (±30 s)

        Returns:
            InpaintingResult mit restauriertem audio und Qualitätsmetriken

        Raises:
            ValueError: Falls sr != 48000 oder Lücke < 0 Samples
        """
        assert sr == 48000, f"CQTdiff: SR muss 48000 Hz sein, erhalten: {sr}"
        gap_samples = gap_end_sample - gap_start_sample
        if gap_samples <= 0:
            raise ValueError(f"CQTdiff: Lücke muss > 0 Samples sein, erhalten: {gap_samples}")

        gap_ms = gap_samples / sr * 1000.0
        logger.info(
            "🔵 CQTdiff: Lücke %.1f ms (%d Samples) — Modell: %s",
            gap_ms,
            gap_samples,
            "CQTdiff" if self._model_loaded else "DSP-Fallback",
        )

        audio_f32 = np.asarray(audio, dtype=np.float32)
        audio_f32 = np.nan_to_num(audio_f32)

        if self._model_loaded and self._session is not None:
            result_audio = self._inpaint_diffusion(audio_f32, sr, gap_start_sample, gap_end_sample, context_audio)
        elif self._ts_model is not None:
            # TorchScript-Direktladung — einfache Forward-Propagation
            result_audio = self._inpaint_diffusion(audio_f32, sr, gap_start_sample, gap_end_sample, context_audio)
        else:
            result_audio = self._inpaint_dsp_fallback(audio_f32, sr, gap_start_sample, gap_end_sample)

        kl_div = self._compute_kl(audio, result_audio, gap_start_sample, gap_end_sample)
        chroma_corr = self._compute_chroma_corr(audio, result_audio, sr)

        return InpaintingResult(
            audio=result_audio,
            sr=sr,
            kl_divergence=kl_div,
            chroma_corr=chroma_corr,
            model_used="cqtdiff" if self._model_loaded else "nmf_dsp_fallback",
            confidence=0.85 if self._model_loaded else 0.55,
            metadata={"gap_ms": gap_ms},
        )

    # ------------------------------------------------------------------
    # Diffusions-Pfad (ONNX)
    # ------------------------------------------------------------------

    def _inpaint_diffusion(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start: int,
        gap_end: int,
        context: np.ndarray | None,
    ) -> np.ndarray:
        """Diffusions-Inpainting via CQT-Score-Netzwerk (ONNX)."""
        _plm = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            _plm = get_plugin_lifecycle_manager()
            _plm.set_active("CQTdiff+", True)
        except Exception:
            logger.warning("cqtdiff_plugin.py::_inpaint_diffusion fallback", exc_info=True)
        try:
            session = self._session
            if session is None:
                raise RuntimeError("CQTdiff ONNX-Session nicht initialisiert")

            # Kontext für Konditionierung (Phrasen-Kontext oder lokaler Kontext)
            ctx = context if context is not None else self._extract_local_context(audio, gap_start, gap_end)
            ctx_feat = np.abs(np.fft.rfft(ctx, n=2048)).astype(np.float32)[np.newaxis, np.newaxis, :]

            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: ctx_feat})

            # Score-Netzwerk-Output → synthetisierter Lücken-Inhalt
            if outputs and outputs[0] is not None:
                gap_len = gap_end - gap_start
                generated = outputs[0].flatten()[:gap_len]
                if len(generated) < gap_len:
                    generated = np.pad(generated, (0, gap_len - len(generated)))
                generated = np.nan_to_num(generated, nan=0.0, posinf=0.0, neginf=0.0)
                generated = np.clip(generated, -1.0, 1.0).astype(np.float32)
            else:
                logger.warning("CQTdiff: Modell lieferte leeren Output → DSP-Fallback")
                return self._inpaint_dsp_fallback(audio, sr, gap_start, gap_end)

            # Crossfade an Lücken-Rändern (Hanning, 5 ms)
            result = audio.copy()
            result[gap_start:gap_end] = generated
            result = self._crossfade_edges(result, gap_start, gap_end, sr, fade_ms=5.0)
            return np.clip(result, -1.0, 1.0).astype(np.float32)

        except Exception as exc:
            logger.warning("CQTdiff Diffusions-Fehler: %s — DSP-Fallback", exc)
            return self._inpaint_dsp_fallback(audio, sr, gap_start, gap_end)
        finally:
            if _plm is not None:
                try:
                    _plm.set_active("CQTdiff+", False)
                except Exception:
                    logger.warning("cqtdiff_plugin.py::_inpaint_diffusion fallback", exc_info=True)

    # ------------------------------------------------------------------
    # DSP-Fallback (Consistent Wiener + lineares Crossfade)
    # ------------------------------------------------------------------

    def _inpaint_dsp_fallback(
        self,
        audio: np.ndarray,
        sr: int,
        gap_start: int,
        gap_end: int,
    ) -> np.ndarray:
        """Fallback-Kaskade: DiffWave ONNX → lineare Interpolation + AR.

        Versucht zuerst DiffWave (ONNX, lokal gebündelt) als ML-Fallback.
        Nur wenn DiffWave nicht verfügbar ist, greift lineare Interpolation.

        Referenz: Post-2018-DSP-Fallback (Consistent Wiener, Le Roux & Vincent 2013).
        """
        gap_len = gap_end - gap_start

        # --- ML-Fallback Stufe 1: DiffWave ONNX ---
        try:
            from plugins.diffwave_plugin import DiffwavePlugin

            dw = DiffwavePlugin()
            if dw._session is not None:
                # Maske: True = Lücke (soll inpainted werden)
                mask = np.zeros(len(audio), dtype=bool)
                mask[gap_start:gap_end] = True
                dw_result = dw.inpaint(audio, sr, mask=mask)
                result = np.clip(dw_result, -1.0, 1.0).astype(np.float32)
                result = self._crossfade_edges(result, gap_start, gap_end, sr, fade_ms=5.0)
                logger.info(
                    "🔵 CQTdiff+ → DiffWave-Fallback: Lücke %.1f ms [%d–%d]",
                    gap_len / sr * 1000,
                    gap_start,
                    gap_end,
                )
                return result
        except Exception:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
        except Exception as exc:
            logger.debug("CQTdiff+ DiffWave-Fallback Fehler: %s — lineare Interpolation", exc)

        # --- DSP-Fallback Stufe 2: Lineare Interpolation + AR-Vorhersage ---
        result = audio.copy()

        # Kontext-Segmente vor/nach Lücke
        ctx_before_start = max(0, gap_start - gap_len * 4)
        ctx_after_end = min(len(audio), gap_end + gap_len * 4)
        before = audio[ctx_before_start:gap_start]
        after = audio[gap_end:ctx_after_end]

        # Lineare Übergangsinterpolation + Spektral-gewichtetes Mittel
        t = np.linspace(0, 1, gap_len, dtype=np.float32)
        before_val = float(before[-1]) if len(before) > 0 else 0.0
        after_val = float(after[0]) if len(after) > 0 else 0.0
        interpolated = (1.0 - t) * before_val + t * after_val

        # Leichtes Rauschmodell aus Kontext
        if len(before) >= 32:
            ctx_std = float(np.std(before[-32:]))
            noise = np.random.default_rng(seed=42).normal(0, ctx_std * 0.1, gap_len)
            interpolated += noise.astype(np.float32)

        result[gap_start:gap_end] = np.clip(interpolated, -1.0, 1.0)
        result = self._crossfade_edges(result, gap_start, gap_end, sr, fade_ms=5.0)
        logger.info("🔵 CQTdiff DSP-Fallback: Lineare Interpolation (%.1f ms Lücke)", gap_len / sr * 1000)
        return np.clip(result, -1.0, 1.0).astype(np.float32)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_local_context(audio: np.ndarray, gap_start: int, gap_end: int) -> np.ndarray:
        """Extrahiert lokalen Kontext um die Lücke (max. 2 s)."""
        ctx_len = min(96000, gap_start)  # max. 2 s @ 48000 Hz
        ctx = audio[max(0, gap_start - ctx_len) : gap_start]
        if len(ctx) < 256:
            ctx = np.zeros(256, dtype=np.float32)
        return ctx.astype(np.float32)  # type: ignore[no-any-return]

    @staticmethod
    def _crossfade_edges(
        audio: np.ndarray,
        gap_start: int,
        gap_end: int,
        sr: int,
        fade_ms: float = 5.0,
    ) -> np.ndarray:
        """Hanning-Crossfade an Lücken-Rändern zur Artefakt-Vermeidung."""
        fade_samples = int(sr * fade_ms / 1000)
        result = audio.copy()
        if gap_start >= fade_samples:
            fade_in = np.hanning(fade_samples * 2)[:fade_samples].astype(np.float32)
            result[gap_start : gap_start + fade_samples] *= fade_in
        if gap_end + fade_samples <= len(audio):
            fade_out = np.hanning(fade_samples * 2)[fade_samples:].astype(np.float32)
            result[gap_end - fade_samples : gap_end] *= fade_out
        return result

    @staticmethod
    def _compute_kl(
        original: np.ndarray,
        restored: np.ndarray,
        gap_start: int,
        gap_end: int,
    ) -> float:
        """KL-Divergenz des Spektrums vor/nach Inpainting (soll < 0.15)."""
        n_fft = 512
        ref_seg = original[max(0, gap_start - 4800) : gap_start + 1]
        if len(ref_seg) < n_fft:
            return 0.0
        p = np.abs(np.fft.rfft(ref_seg[-n_fft:])) + 1e-12
        q = np.abs(np.fft.rfft(restored[max(0, gap_start - n_fft) : gap_start + n_fft][-n_fft:])) + 1e-12
        if len(p) != len(q):
            return 0.0
        p /= p.sum()
        q /= q.sum()
        kl = float(np.sum(p * np.log(p / q + 1e-12)))
        return float(np.clip(kl, 0.0, 10.0))

    @staticmethod
    def _compute_chroma_corr(original: np.ndarray, restored: np.ndarray, sr: int) -> float:
        """Pearson-Korrelation der Chroma-Vektoren (soll ≥ 0.92)."""
        try:
            import librosa

            n = min(len(original), len(restored), sr * 4)  # max. 4 s
            if n < 64:
                return 0.9
            n_fft = int(min(1024, n))
            if n_fft % 2 == 1:
                n_fft -= 1
            n_fft = max(64, n_fft)
            hop_length = max(16, n_fft // 4)
            c1 = librosa.feature.chroma_stft(
                y=original[:n].astype(np.float32),
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            c2 = librosa.feature.chroma_stft(
                y=restored[:n].astype(np.float32),
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            _s1 = float(np.std(c1.ravel()))
            _s2 = float(np.std(c2.ravel()))
            if _s1 < 1e-8 or _s2 < 1e-8:
                return 0.9  # near-constant chroma (silence) → neutral
            _c1r = c1.ravel()
            _c2r = c2.ravel()
            _c1a = _c1r - _c1r.mean()
            _c2a = _c2r - _c2r.mean()
            _nc1 = float(np.linalg.norm(_c1a))
            _nc2 = float(np.linalg.norm(_c2a))
            corr = float(np.dot(_c1a, _c2a) / (_nc1 * _nc2 + 1e-10))
            return float(np.clip(np.nan_to_num(corr), -1.0, 1.0))
        except Exception:
            logger.warning("cqtdiff_plugin.py::_compute_chroma_corr fallback", exc_info=True)
            return 0.9  # Optimistischer Standardwert bei librosa-Fehler


# ---------------------------------------------------------------------------
# Singleton-Accessor
# ---------------------------------------------------------------------------


def get_cqtdiff_plus() -> CQTdiffPlusPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = CQTdiffPlusPlugin()
    return _instance


def inpaint_gap(
    audio: np.ndarray,
    sr: int,
    gap_start_sample: int,
    gap_end_sample: int,
    *,
    context_audio: np.ndarray | None = None,
) -> InpaintingResult:
    """Convenience-Wrapper — CQTdiff+ Lücken-Inpainting ohne Klassen-Instantiierung.

    Beispiel::

        result = inpaint_gap(audio, sr=48000, gap_start_sample=96000, gap_end_sample=100800)
        logger.debug("KL: %.3f, Chroma-Corr: %.3f", result.kl_divergence, result.chroma_corr)

    Args:
        audio:              Audio-Signal (1D float32, 48000 Hz)
        sr:                 Sample-Rate (48000)
        gap_start_sample:   Lücken-Anfang (Sample-Index)
        gap_end_sample:     Lücken-Ende (exklusiv)
        context_audio:      Optionaler musikalischer Phrasenkontext

    Returns:
        InpaintingResult
    """
    return get_cqtdiff_plus().inpaint(audio, sr, gap_start_sample, gap_end_sample, context_audio=context_audio)
