"""Harmonic Inpainting Plugin (§v10.300) — DiT-Finetune für gedämpfte Obertöne.

Selbst trainiert: FlowMatchingDiT (201M, 48 kHz) fine-getuned auf Harmonic
Inpainting (train_harmonic_inpainting.py). Rekonstruiert genau die harmonischen
Anteile, die der Denoiser (DFN) gedämpft hat — Rectified-Flow-Einschritt:
    v(x, t) ≈ clean − attenuated  (konstantes Geschwindigkeitsfeld)
    out     = attenuated + v · mask

Compliance-Einbindung (1:1 in bestehende Vorgaben):
  - §v10.19 Feature-Flag-Routing: use_harmonic_inpainting + resolve_model_path()
  - §G101: Aufrufer blended via perceptual_blend() (Bark-Band-Wet)
  - §G88:  Aufrufer deaktiviert ML bei transfer_depth ≥ 5 (DSP-Fallback)
  - §G136: deterministisch — eval(), no_grad(), feste Dämpfung 0.5 statt
           Trainings-Zufall, stabile Chunk-Auswahl
  - ML-Budget + PLM-Registrierung wie alle anderen Plugins; bei Fehler → None
    (Aufrufer nutzt bestehende DSP-Harmonik-Synthese als Fallback)

Kein Netzwerk, keine Downloads — alles lokal.
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Modulebene (nicht unter Lock! §G174) — music_model_flags hat keine Backend-Abhängigkeiten.
try:
    from backend.core.music_model_flags import resolve_model_path, use_harmonic_inpainting

    _FLAGS_AVAILABLE = True
except Exception as _flags_exc:
    _FLAGS_AVAILABLE = False
    resolve_model_path = None  # type: ignore[assignment]
    use_harmonic_inpainting = False
    logger.debug("HarmonicInpainting: music_model_flags nicht ladbar: %s", _flags_exc)

# dit_model-Import ebenfalls auf Modulebene (§G174); kein Backend-Import.
try:
    _DIT_DIR = str(Path(__file__).resolve().parent.parent / "models" / "miipher_dit")
    if _DIT_DIR not in sys.path:
        sys.path.insert(0, _DIT_DIR)
    from dit_model import FlowMatchingDiT  # type: ignore[import]

    _DIT_AVAILABLE = True
except Exception as _dit_exc:
    _DIT_AVAILABLE = False
    FlowMatchingDiT = None  # type: ignore[assignment]
    logger.debug("HarmonicInpainting: dit_model nicht ladbar: %s", _dit_exc)

_lock = threading.Lock()
_inst: HarmonicInpaintingPlugin | None = None

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

_ROOT = Path(__file__).resolve().parent.parent
_SR = 48_000
_CHUNK_SEC = 2.0
_CHUNK_SAMPLES = int(_CHUNK_SEC * _SR)  # 96000 — wie train_harmonic_inpainting.py
_N_FFT = 2048
_HOP = 512
_MAX_ML_SECONDS = 60.0  # Kostenbudget: höchstens 60 s Audio pro Aufruf (§4.5)
_FIXED_ATTENUATION = 0.5  # deterministisch statt np.random.uniform(0.3, 0.6)


def _mask_generate_deterministic(audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Deterministische Variante des HarmonicMaskGenerator (Training §v10.300).

    Returns: (attenuated_audio, inpainting_mask) — beide [T], float32.
    mask: 1.0 = rekonstruieren, 0.0 = unberührt.
    """
    audio = audio.astype(np.float32)
    n_frames = 1 + (len(audio) - _N_FFT) // _HOP
    window = np.hanning(_N_FFT).astype(np.float32)
    specgram = np.zeros((n_frames, _N_FFT // 2 + 1), dtype=np.float32)
    for i in range(n_frames):
        start = i * _HOP
        specgram[i] = np.abs(np.fft.rfft(audio[start : start + _N_FFT] * window))

    median_spec = np.median(specgram, axis=0)
    median_global = float(np.median(median_spec))
    harmonic_mask_freq = np.zeros(_N_FFT // 2 + 1, dtype=bool)
    for freq_bin in range(1, _N_FFT // 2 + 1):
        if median_spec[freq_bin] > median_global * 2.5:
            harmonic_mask_freq[max(0, freq_bin - 2) : freq_bin + 3] = True

    frame_energy = specgram.mean(axis=1)
    loud_frames = frame_energy > np.median(frame_energy)

    mask = np.zeros(len(audio), dtype=np.float32)
    for i in range(n_frames):
        if loud_frames[i]:
            start = i * _HOP
            end = min(start + _N_FFT, len(audio))
            mask[start:end] = _FIXED_ATTENUATION * window[: end - start]

    attenuated = audio * (1.0 - mask)
    inpainting_mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return attenuated.astype(np.float32), inpainting_mask


class HarmonicInpaintingPlugin:
    """FlowMatchingDiT-Finetune für harmonische Rekonstruktion (§v10.300)."""

    _BUDGET_NAME: str = "HarmonicInpaintingDiT"
    _BUDGET_SIZE_GB: float = 1.8  # 201M fp32 (~0.8 GB) + Aktivierungen

    def __init__(self) -> None:
        self._model: Any = None
        self._loaded: bool = False
        self._try_load()

    def _try_load(self) -> None:
        if not _TORCH_AVAILABLE:
            logger.warning("HarmonicInpainting: torch fehlt — DSP-Ersatzpfad aktiv.")
            return
        if not _FLAGS_AVAILABLE or not use_harmonic_inpainting:
            logger.info("HarmonicInpainting: Feature-Flag aus — DSP-Ersatzpfad.")
            return
        _p = resolve_model_path("harmonic_inpainting") if resolve_model_path is not None else None
        if _p is None or not Path(_p).exists():
            logger.info("HarmonicInpainting: Checkpoint fehlt — DSP-Ersatzpfad.")
            return

        # ML-Memory-Budget VOR dem Laden (§5.1 OOM-Schutz)
        try:
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("HarmonicInpainting: ML-Budget erschöpft — DSP-Ersatzpfad.")
                return
        except ImportError:
            pass

        if not _DIT_AVAILABLE:
            logger.warning("HarmonicInpainting: dit_model fehlt — DSP-Ersatzpfad.")
            return
        try:
            model = FlowMatchingDiT(dropout=0.0)  # Modulebene importiert (§G174)
            ckpt = torch.load(str(_p), map_location="cpu", weights_only=True)
            sd = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(sd)
            model.eval()
            self._model = model
            self._loaded = True
            logger.info(
                "HarmonicInpainting geladen: %s (val_loss=%s, %.1f MB)",
                Path(_p).name,
                ckpt.get("val_loss", "?") if isinstance(ckpt, dict) else "?",
                Path(_p).stat().st_size / 1e6,
            )

            # PLM-Registrierung (§4.6b)
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _reg_plm(
                    self._BUDGET_NAME,
                    size_gb=self._BUDGET_SIZE_GB,
                    unload_fn=self.unload,
                )
            except Exception:
                pass
        except Exception as exc:
            logger.warning("HarmonicInpainting Ladefehler: %s — DSP-Ersatzpfad.", exc)
            self._model = None
            self._loaded = False

    def unload(self) -> None:
        """Entlädt das Modell (PLM-Eviction-Callback)."""
        self._model = None
        self._loaded = False
        try:
            from backend.core.ml_memory_budget import release as _release

            _release(self._BUDGET_NAME)
        except Exception:
            pass
        logger.debug("HarmonicInpainting entladen")

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _inpaint_chunk(self, chunk: np.ndarray) -> np.ndarray | None:
        """Rectified-Flow-Einschritt auf einem 2s-Chunk [T].

        Returns rekonstruiertes Audio [T] oder None bei Fehler.
        """
        if self._model is None:
            return None
        if len(chunk) != _CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, _CHUNK_SAMPLES - len(chunk)), mode="reflect")[:_CHUNK_SAMPLES]
        peak = float(np.abs(chunk).max() + 1e-10)
        if peak < 1e-8:
            return None
        norm = (chunk / peak).astype(np.float32)
        attenuated, mask = _mask_generate_deterministic(norm)

        try:
            x = torch.from_numpy(attenuated).unsqueeze(0).unsqueeze(-1)  # [1, T, 1]
            t_val = torch.tensor([0.1])
            with torch.no_grad():
                v = self._model(x, t_val)  # [1, T, 1]
            v_np = v.squeeze().cpu().numpy().astype(np.float32)
            out = attenuated + (v_np * mask)  # nur Inpainting-Regionen
            return cast(np.ndarray | None, ((np.clip(out, -1.0, 1.0) * peak).astype(np.float32)[: len(chunk)]))
        except Exception as exc:
            logger.debug("HarmonicInpainting Chunk-Fehler: %s", exc)
            return None

    def enhance(self, audio_mono: np.ndarray, sr: int) -> np.ndarray | None:
        """Rekonstruiert gedämpfte Obertöne (mono, 48 kHz).

        Verarbeitet deterministisch die am stärksten betroffenen Fenster
        (Budget: _MAX_ML_SECONDS), Rest bleibt unverändert (§4.5 Kostenkontrolle).

        Returns: verbessertes Audio [T] oder None (→ DSP-Fallback beim Aufrufer).
        """
        if not self._loaded or audio_mono is None or len(audio_mono) < _CHUNK_SAMPLES // 2:
            return None
        try:
            audio = np.asarray(audio_mono, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sr != _SR:
                import librosa

                audio = librosa.resample(audio, orig_sr=sr, target_sr=_SR).astype(np.float32)

            n_chunks = int(np.ceil(len(audio) / _CHUNK_SAMPLES))
            if n_chunks <= 0:
                return None

            # Deterministische Chunk-Auswahl nach Masken-Energie (§G136)
            order = list(range(n_chunks))
            energies: list[float] = []
            for i in order:
                start = i * _CHUNK_SAMPLES
                seg = audio[start : start + _CHUNK_SAMPLES]
                if len(seg) < _CHUNK_SAMPLES:
                    seg = np.pad(seg, (0, _CHUNK_SAMPLES - len(seg)), mode="reflect")
                _, m = _mask_generate_deterministic(seg)
                energies.append(float(np.mean(m)))
            order.sort(key=lambda i: -energies[i])  # stabil: absteigende Energie

            max_chunks = int(_MAX_ML_SECONDS / _CHUNK_SEC)
            chosen = sorted(order[:max_chunks])

            out = audio.copy()
            for i in chosen:
                start = i * _CHUNK_SAMPLES
                seg = audio[start : start + _CHUNK_SAMPLES]
                res = self._inpaint_chunk(seg)
                if res is not None:
                    out[start : start + _CHUNK_SAMPLES] = res[: len(seg)]

            if sr != _SR:
                import librosa

                out = librosa.resample(out, orig_sr=_SR, target_sr=sr).astype(np.float32)
            return cast(np.ndarray | None, (np.clip(out, -1.0, 1.0)))
        except Exception as exc:
            logger.debug("HarmonicInpainting enhance Fehler: %s", exc)
            return None


def get_harmonic_inpainting_plugin() -> HarmonicInpaintingPlugin:
    """Singleton-Zugriff (thread-sicher)."""
    global _inst
    with _lock:
        if _inst is None:
            _inst = HarmonicInpaintingPlugin()
        return _inst
