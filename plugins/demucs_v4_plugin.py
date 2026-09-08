"""
DemucsV4Plugin — Stem-Separation via htdemucs_6s.onnx (lokal).
Kein Docker, kein Netzwerk.

Referenz: Défossez et al. (2023) Hybrid Transformers for Music Source Separation.
ONNX-Interface htdemucs_6s.onnx:
  IN:  input[1,2,343980]  (4 Sekunden Stereo @ 48 kHz)
       x[1,4,2048,336]    (Spectrogramm-Konditionierung, intern durch HPSS gefüllt)
  OUT: add_67[1,6,2,343980]  (6 Stems: drums/bass/other/vocals/guitar/piano)
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_lock = threading.Lock()
_inst_holder: list[DemucsV4Plugin | None] = [None]

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_PATH = os.path.join(_ROOT, "models", "demucs", "htdemucs_6s.onnx")

# Modell-Konstanten
_SR = 44_100  # Demucs arbeitet mit 44.1 kHz
_CHUNK = 343_980  # Genau 1 Modell-Chunk (~ 7.8 s @ 44.1 kHz)
_SPEC_FRAMES = 336
_SPEC_BINS = 2048
_SPEC_CH = 4
_STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]


class DemucsV4Plugin:
    """htdemucs_6s Stem-Separation (ONNX) mit HPSS-DSP-Fallback."""

    def __init__(self, model_path: str | None = None, root: str | None = None) -> None:
        p = model_path or _MODEL_PATH
        if root:
            p = os.path.join(root, "models", "demucs", "htdemucs_6s.onnx")
        self._session: Any = None
        self._model_path = p
        self._try_load()

    def _try_load(self) -> None:
        if not os.path.exists(self._model_path):
            logger.warning("Demucs-Modell fehlt: %s — DSP-Fallback aktiv.", self._model_path)
            return
        # §Fix 2026-09-08 (SOTA-Root-Cause): Das frühere Manifest-Gate
        # (models/manifest.json, gitignored) deaktivierte die Demucs-Stufe
        # STILL (experimental=True) — Verstoß §V6 und §V7. Produktions-Modelle
        # werden jetzt standardmäßig geladen; ein expliziter, dokumentierter
        # Opt-out ersetzt das stille Gate:
        #     AURIK_DISABLE_HTDEMUCS_6S=1 → DSP-Fallback (Debug/Notbetrieb).
        if os.environ.get("AURIK_DISABLE_HTDEMUCS_6S") == "1":
            logger.warning(
                "HTDemucs 6s: AURIK_DISABLE_HTDEMUCS_6S=1 gesetzt — "
                "ONNX-Session nicht geladen, DSP-Fallback aktiv."
            )
            return
        try:
            import onnxruntime as ort  # pylint: disable=import-outside-toplevel

            try:
                from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                    try_allocate as _try_alloc,
                )

                if not _try_alloc("DemucsV4", size_gb=0.12):
                    try:
                        from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                            release as _rel2,
                        )

                        _rel2("DemucsV4")
                    except Exception:
                        logger.warning("demucs_v4_plugin.py::_try_load fallback", exc_info=True)
                    if not _try_alloc("DemucsV4", size_gb=0.12):
                        logger.warning("DemucsV4: ML-Budget erschöpft — HPSS-Fallback.")
                        return
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            try:
                from backend.core.ml_device_manager import (  # pylint: disable=import-outside-toplevel
                    get_ort_providers as _get_prov,
                )

                _providers = _get_prov("DemucsV4")
            except Exception:
                _providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(self._model_path, sess_options=opts, providers=_providers)
            logger.info("Demucs htdemucs_6s ONNX geladen: %s", self._model_path)
            try:
                from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                    register_plugin as _reg_plm,
                )

                _reg_plm("DemucsV4", size_gb=0.12, unload_fn=lambda s=self: setattr(s, "_session", None))  # type: ignore[misc]
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("Demucs ONNX-Ladefehler: %s — DSP-Fallback aktiv.", exc)
            try:
                from backend.core.ml_memory_budget import (  # pylint: disable=import-outside-toplevel
                    release as _rel,
                )

                _rel("DemucsV4")
            except Exception as _exc:
                logger.debug("Operation failed (non-critical): %s", _exc)

    # ── Public API ───────────────────────────────────────────────────────────

    def separate(self, audio: np.ndarray, sr: int, prefer_mdx23c: bool = True) -> dict[str, np.ndarray]:
        """Stem-Separation: gibt Dict stem→audio zurück (selbe SR wie Eingang).

        Args:
            audio: float32 stereo [n,2] oder mono [n] (muss 48000 Hz sein).
            sr:    Sample-Rate des Eingangs (muss 48000 Hz sein).

        Returns:
            Dict mit Schlüsseln "vocals", "drums", "bass", "other", "guitar", "piano".

        Priority:
            - prefer_mdx23c=True: MDX23C (Kim_Vocal_2) → HTDemucs 6s ONNX → HPSS-DSP
            - prefer_mdx23c=False: HTDemucs 6s ONNX → HPSS-DSP
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # Normalize to (2, N) channels-first — required by MDX23C/HTDemucs.
        # UV3 sends (2, N); (N, 2) samples-first is transposed; unexpected layouts fallback to first row.
        if audio.ndim == 1:
            audio = np.stack([audio, audio], axis=0)  # (N,) → (2, N)
        elif audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            pass  # already (2, N) channels-first — correct for MDX23C
        elif audio.ndim == 2 and audio.shape[1] == 2 and audio.shape[0] != 2:
            audio = audio.T  # (N, 2) → (2, N)
        elif audio.ndim == 2 and audio.shape[1] != 2:
            audio = np.stack([audio[0], audio[0]], axis=0)  # (C, N) unexpected → duplicate ch0

        # Optional Primary: MDX23C (Kim_Vocal_2) — production-grade vocal separation (§4.4 spec)
        if prefer_mdx23c:
            try:
                from plugins.mdx23c_plugin import (  # pylint: disable=import-outside-toplevel
                    separate_stems as _mdx_stems,
                )

                mdx_result = _mdx_stems(audio, sr)
                if mdx_result and "vocals" in mdx_result:
                    logger.info("DemucsV4: MDX23C primary path used (Kim_Vocal_2).")
                    return mdx_result
            except Exception as exc:
                logger.warning("DemucsV4: MDX23C primary failed (%s) — HTDemucs/HPSS fallback.", exc)

        # Fallback 1: HTDemucs 6s ONNX (if loaded)
        if self._session is not None:
            _plm_dmu = None
            try:
                from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                    get_plugin_lifecycle_manager as _get_plm_fn,
                )

                _plm_dmu = _get_plm_fn()
                _plm_dmu.set_active("DemucsV4", True)
            except Exception as _exc:
                logger.debug("DemucsV4: PLM set_active failed: %s", _exc)
            try:
                return self._infer_onnx(audio, sr)
            finally:
                if _plm_dmu is not None:
                    try:
                        _plm_dmu.set_active("DemucsV4", False)
                    except Exception as _exc:
                        logger.debug("DemucsV4: PLM unset_active failed: %s", _exc)

        # Fallback 2: HPSS-DSP
        return self._hpss_fallback(audio, sr)

    def separate_vocals(
        self,
        audio: np.ndarray,
        sr: int,
        prefer_mdx23c: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Gibt (vocals, instruments) zurück (Shortcut für 2-Stem-Betrieb)."""
        stems = self.separate(audio, sr, prefer_mdx23c=prefer_mdx23c)
        vocals = stems.get("vocals", audio)
        non_vocals = ["drums", "bass", "other", "guitar", "piano"]
        inst_arrays = [stems[k] for k in non_vocals if k in stems]
        instruments = np.mean(inst_arrays, axis=0) if inst_arrays else audio - vocals
        return vocals, instruments

    def process(self, audio, sr):
        """Backwards-Compatibility-Alias fuer separate() - Standard-Plugin-Interface."""
        return self.separate(audio, sr)

    # ── Internal ─────────────────────────────────────────────────────────────

    def _resample(self, audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        if sr_from == sr_to:
            return audio
        from scipy.signal import resample_poly  # pylint: disable=import-outside-toplevel

        g = math.gcd(sr_from, sr_to)
        up, down = sr_to // g, sr_from // g
        # §2.51 Stereo-Axis-Invariante: Plugin-Kontrakt ist channels-first (2, N).
        # §Fix 2026-09-08: Vorher columns-Zugriff (audio[:, 0]) — für (2, N)-
        # Eingaben lieferte das 2-Sample-Arrays → 2×2-Stems (stiller Müll).
        if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
            ch0 = resample_poly(audio[0], up, down).astype(np.float32)
            ch1 = resample_poly(audio[1], up, down).astype(np.float32)
            n = min(len(ch0), len(ch1))
            return np.stack([ch0[:n], ch1[:n]], axis=0)  # (2, n) channels-first
        left = resample_poly(audio[:, 0], up, down).astype(np.float32)
        right = resample_poly(audio[:, 1], up, down).astype(np.float32)
        n = min(len(left), len(right))
        return np.stack([left[:n], right[:n]], axis=1)  # type: ignore[no-any-return]

    def _make_spec_cond(self, chunk: np.ndarray) -> np.ndarray:
        """Erstelle Spektrogramm-Konditionierung x[1,4,2048,336] via STFT."""
        win = np.hanning(4096).astype(np.float32)
        hop = _CHUNK // _SPEC_FRAMES

        specs = []
        for ch in [0, 1]:
            s = chunk[ch]  # §2.51 channels-first
            frames = []
            for i in range(_SPEC_FRAMES):
                start = i * hop
                seg = np.zeros(4096, dtype=np.float32)
                end = min(start + 4096, len(s))
                seg[: end - start] = s[start:end]
                f = np.abs(np.fft.rfft(seg * win)[:_SPEC_BINS]).astype(np.float32)
                frames.append(f)
            spec = np.array(frames, dtype=np.float32).T  # [2048, 336]
            specs.append(spec)
        # 4 Kanäle: 2× Magnitude + 2× log-Magnitude
        mag_l, mag_r = specs[0], specs[1]
        log_l = np.log1p(mag_l)
        log_r = np.log1p(mag_r)
        x = np.stack([mag_l, mag_r, log_l, log_r], axis=0)  # [4,2048,336]
        return x[np.newaxis].astype(np.float32)  # type: ignore[no-any-return]  # [1,4,2048,336]

    def _infer_onnx(self, audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
        # §2.51: channels-first (2, N) — Originallänge ist die Sample-Achse
        n_orig = audio.shape[1] if audio.ndim == 2 and audio.shape[0] == 2 else len(audio)
        # Resampling auf Modell-SR
        audio_r = self._resample(audio, sr, _SR)
        n = audio_r.shape[1] if audio_r.ndim == 2 else len(audio_r)

        # Chunked Verarbeitung
        stride = _CHUNK
        n_chunks = max(1, math.ceil(n / stride))
        out_stems = {s: np.zeros_like(audio_r) for s in _STEMS}

        for i in range(n_chunks):
            start = i * stride
            chunk = np.zeros((2, _CHUNK), dtype=np.float32)  # §2.51 channels-first
            end = min(start + _CHUNK, n)
            chunk[:, : end - start] = audio_r[:, start:end]

            # Eingaben
            inp = chunk[np.newaxis].astype(np.float32)  # [1,2,343980]
            x = self._make_spec_cond(chunk)  # [1,4,2048,336]

            try:
                outputs = self._session.run(None, {"input": inp, "x": x})
                # output add_67: [1,6,2,343980]
                result = None
                for o in outputs:
                    if hasattr(o, "shape") and o.ndim == 4 and o.shape[1] == 6:
                        result = o
                        break
                if result is None:
                    result = outputs[-1] if outputs else None

                if result is not None and result.shape[1] == len(_STEMS):
                    for si, name in enumerate(_STEMS):
                        seg = np.asarray(result)[0, si, :, : end - start]  # [2, n]
                        out_stems[name][:, start:end] += seg[:, : end - start]
                else:
                    raise ValueError(f"Unerwartetes Output-Shape: {[o.shape for o in outputs]}")
            except Exception as exc:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
                logger.debug("Demucs Chunk %d Fehler: %s — DSP.", i, exc)
                fb = self._hpss_fallback(chunk, _SR)
                for k, v in fb.items():
                    out_stems[k][:, start:end] += v[:, : end - start]

        # Rückresampling auf Original-SR
        if sr != _SR:
            out_stems = {k: self._resample(v, _SR, sr) for k, v in out_stems.items()}
        # Auf Originallänge kürzen (§2.51: channels-first → Achse 1 kürzen)
        return {k: v[:, :n_orig] for k, v in out_stems.items()}

    @staticmethod
    def _hpss_fallback(audio: np.ndarray, _sr: int) -> dict[str, np.ndarray]:
        """HPSS-basierter Stem-Fallback bei fehlendem Modell (channels-first, §2.51)."""
        try:
            import librosa  # pylint: disable=import-outside-toplevel

            if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
                mono = audio[0]  # channels-first
            elif audio.ndim == 2:
                mono = audio[:, 0]  # channels-last
            else:
                mono = audio
            H, P = librosa.effects.hpss(mono)  # type: ignore[attr-defined]
            H_st = np.stack([H, H], axis=0)
            P_st = np.stack([P, P], axis=0)
            res = audio - H_st - P_st
            return {
                "vocals": H_st,
                "drums": P_st,
                "bass": P_st * 0.5,
                "other": res,
                "guitar": res * 0.5,
                "piano": res * 0.3,
            }
        except Exception:
            result = {k: audio.copy() for k in _STEMS}
            return result


# ── Singleton ────────────────────────────────────────────────────────────────


def get_demucs_plugin() -> DemucsV4Plugin:
    """Thread-sicherer Singleton (Double-Checked Locking)."""
    if _inst_holder[0] is None:
        with _lock:
            if _inst_holder[0] is None:
                _inst_holder[0] = DemucsV4Plugin()
    return _inst_holder[0]  # type: ignore[return-value]


def separate_stems(audio: np.ndarray, sr: int) -> dict[str, np.ndarray]:
    """Convenience-Wrapper: Stem-Separation via Demucs/HPSS."""
    return get_demucs_plugin().separate(audio, sr)


def separate_vocals_instruments(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Convenience-Wrapper: (vocals, instruments) trennen."""
    return get_demucs_plugin().separate_vocals(audio, sr)


# Convenience-Alias
def run_demucs(audio: np.ndarray, sr: int = 48000) -> dict:
    """Alias für separate_stems."""
    return separate_stems(audio, sr)
