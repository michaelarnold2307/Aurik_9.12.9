"""WhisperDenoiserPlugin — DEPRECATED (Rev. 2026-08-16).

⚠️ Nicht mehr Teil der Produktions-NR-Kette: Denoising tragen DeepFilterNet v3 II /
SGMSE+ / OMLSA gemäß Spec 04 (§7.11, phase_66). Dieses Plugin bleibt nur für
A/B-Vergleiche über music_model_flags.use_whisper_denoiser (= False) ladbar.

Historisch: Whisper-gesteuertes Musik-Denoising (§v10.20).
Selbst trainiert (MUSDB18-HQ, 48 kHz): Whisper-tiny (frozen, 39M) als
Feature-Extraktor + ConditionedUNet (2M) + LightweightDecoder (2M).
Training: AurikLoss = 0.7 × MSE(complex STFT) + 0.3 × BarkLoss (psychoakustisch).

Compliance-Einbindung:
  - §G100/§G101: Ausgabe wird im Aufrufer via perceptual_blend geblendet
    (hier nur sauberer ML-Adapter, kein skalarer Blend)
  - §G104: JND-Gate greift zentral in UV3 nach der Phase
  - §G88: DSP-Fallback (OMLSA) übernimmt, wenn Modell nicht verfügbar
  - §G136: deterministische Inferenz (eval, no_grad, fester Seed-freier Pfad)
  - PLM-registriert, ML-Memory-Budget, lazy load

Interface-Kompatibilität mit DeepFilterNetV3IIPlugin.enhance():
    enhance(audio_mono, sr, energy_bias_db=None) -> np.ndarray
"""

from __future__ import annotations

import importlib.util
import logging
import os
import threading
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# Modulebene (§G174): Flags ohne Backend-Abhängigkeiten.
try:
    from backend.core.music_model_flags import resolve_model_path, use_whisper_denoiser

    _FLAGS_AVAILABLE = True
except Exception as _flags_exc:
    _FLAGS_AVAILABLE = False
    resolve_model_path = None  # type: ignore[assignment]
    use_whisper_denoiser = False
    logger.debug("WhisperDenoiser: music_model_flags nicht ladbar: %s", _flags_exc)

_lock = threading.Lock()
_inst: WhisperDenoiserPlugin | None = None

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]

_ROOT = Path(__file__).resolve().parent.parent
_SR = 48_000
_CHUNK_SEC = 4.0
_CHUNK_SAMPLES = int(_CHUNK_SEC * _SR)


def _load_trainer_module() -> Any:
    """Lädt die Modellklassen aus dem Trainings-Skript (kein Re-Import von torch-Code)."""
    _trainer_path = _ROOT / "scripts" / "train_whisper_denoiser.py"
    if not _trainer_path.exists():
        return None
    try:
        _spec = importlib.util.spec_from_file_location("aurik_whisper_denoiser_trainer", str(_trainer_path))
        if _spec is None or _spec.loader is None:
            return None
        _mod = importlib.util.module_from_spec(_spec)
        # Offline-Erzwingung: Whisper-tiny muss lokal gecacht sein (kein Download)
        _prev_hf = os.environ.get("HF_HUB_OFFLINE")
        _prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            _spec.loader.exec_module(_mod)
        finally:
            if _prev_hf is None:
                os.environ.pop("HF_HUB_OFFLINE", None)
            else:
                os.environ["HF_HUB_OFFLINE"] = _prev_hf
            if _prev_tf is None:
                os.environ.pop("TRANSFORMERS_OFFLINE", None)
            else:
                os.environ["TRANSFORMERS_OFFLINE"] = _prev_tf
        return _mod
    except Exception as _exc:
        logger.debug("WhisperDenoiser: Trainer-Modul nicht ladbar: %s", _exc)
        return None


# Modulebene laden (§G174: kein Import unter Lock). Das Trainings-Skript
# importiert nur leichte Abhängigkeiten (transformers wird erst in
# WhisperFeatureExtractor.__init__ lazy geladen).
_TRAINER = _load_trainer_module()
if _TRAINER is None:
    logger.debug("WhisperDenoiser: Trainer-Modul fehlt — Plugin bleibt inaktiv")


class WhisperDenoiserPlugin:
    """Whisper-gesteuertes Denoising für Musik-Stems (Mono, 48 kHz)."""

    _BUDGET_NAME: str = "WhisperDenoiser"
    _BUDGET_SIZE_GB: float = 0.6  # Whisper-tiny ~150 MB + UNet/Decoder + Aktivierungen

    def __init__(self) -> None:
        self._model: Any = None
        self._trainer: Any = None
        self._try_load()

    def _try_load(self) -> None:
        if not _TORCH_AVAILABLE:
            logger.warning("WhisperDenoiser: torch fehlt — Plugin inaktiv (OMLSA-Fallback)")
            return
        if not _FLAGS_AVAILABLE or not use_whisper_denoiser:
            logger.info("WhisperDenoiser: Feature-Flag use_whisper_denoiser=False — inaktiv")
            return
        _ckpt = resolve_model_path("whisper_denoiser") if resolve_model_path is not None else None
        if _ckpt is None or not Path(_ckpt).exists():
            logger.warning("WhisperDenoiser: Checkpoint fehlt — Plugin inaktiv")
            return
        try:
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                logger.info("WhisperDenoiser: ML-Budget erschöpft — Plugin inaktiv")
                return
        except ImportError:
            pass
        try:
            if _TRAINER is None:
                logger.warning("WhisperDenoiser: Trainer-Modul fehlt — Plugin inaktiv")
                return
            # transformers WhisperModel aus lokalem Cache (offline erzwungen beim Modul-Load)
            _model = _TRAINER.WhisperDenoiser(device="cpu")
            _ckpt_data = torch.load(str(_ckpt), map_location="cpu", weights_only=True)
            _model.unet.load_state_dict(_ckpt_data["unet_state_dict"])
            _model.decoder.load_state_dict(_ckpt_data["decoder_state_dict"])
            _model.eval()
            self._model = _model
            self._trainer = _TRAINER
            logger.info(
                "WhisperDenoiser geladen: %s (val_loss=%.4f, epoch=%s)",
                Path(_ckpt).name,
                float(_ckpt_data.get("val_loss", float("nan"))),
                _ckpt_data.get("epoch"),
            )
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin

                register_plugin(
                    self._BUDGET_NAME,
                    size_gb=self._BUDGET_SIZE_GB,
                    unload_fn=self.unload,
                )
            except Exception:
                pass
        except Exception as _exc:
            logger.warning("WhisperDenoiser: Ladefehler — OMLSA-Fallback aktiv: %s", _exc)
            self._model = None

    def unload(self) -> None:
        self._model = None
        try:
            from backend.core.ml_memory_budget import release

            release(self._BUDGET_NAME)
        except Exception:
            pass
        logger.debug("WhisperDenoiser entladen")

    @property
    def is_loaded(self) -> bool:
        """True wenn Modell geladen und einsatzbereit (§G88-Gate für Aufrufer)."""
        return self._model is not None

    def enhance(
        self,
        audio_mono: np.ndarray,
        sr: int,
        energy_bias_db: float | None = None,
    ) -> np.ndarray:
        """Denoised mono audio; wirft Exception bei Nichtverfügbarkeit (Aufrufer → OMLSA).

        energy_bias_db wird für Interface-Kompatibilität akzeptiert (DFN-Parameter),
        hat aber keine Wirkung auf das Whisper-Modell (BarkLoss-optimiert).
        """
        if self._model is None:
            raise RuntimeError("WhisperDenoiser nicht geladen")
        if energy_bias_db is not None:
            logger.debug("WhisperDenoiser: energy_bias_db=%.1f ignoriert (Interface-Kompatibilität)", energy_bias_db)

        _x = np.asarray(audio_mono, dtype=np.float32)
        if _x.ndim == 2:
            _x = _x.mean(axis=0)
        if sr != _SR:
            try:
                import librosa

                _x = librosa.resample(_x, orig_sr=sr, target_sr=_SR)
            except Exception as _exc:
                raise RuntimeError(f"Resampling fehlgeschlagen: {_exc}") from _exc
        _orig_len = len(_x)

        _chunks: list[np.ndarray] = []
        for _start in range(0, _orig_len, _CHUNK_SAMPLES):
            _c = _x[_start : _start + _CHUNK_SAMPLES]
            if len(_c) < _CHUNK_SAMPLES:
                _c = np.pad(_c, (0, _CHUNK_SAMPLES - len(_c)), mode="reflect")
            _chunks.append(_c)

        _out_parts: list[np.ndarray] = []
        with torch.no_grad():
            for _c in _chunks:
                _peak = float(np.abs(_c).max() + 1e-10)
                _c_norm = (_c / _peak).astype(np.float32)
                _t = torch.from_numpy(_c_norm).unsqueeze(0)  # [1, T]
                _clean = self._model(_t)  # [1, T]
                _out_parts.append(_clean.squeeze(0).numpy().astype(np.float32) * _peak)

        _out = np.concatenate(_out_parts)[:_orig_len]
        _out = np.nan_to_num(_out, nan=0.0, posinf=0.0, neginf=0.0)
        return cast(np.ndarray, (np.clip(_out, -1.0, 1.0).astype(np.float32)))


def get_whisper_denoiser_plugin() -> WhisperDenoiserPlugin:
    """Singleton mit Lazy Load (§4.6b PLM-kompatibel)."""
    global _inst
    with _lock:
        if _inst is None:
            _inst = WhisperDenoiserPlugin()
        return _inst
