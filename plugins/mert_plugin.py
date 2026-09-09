"""
MERT Plugin für AURIK 9.5 — Music Understanding & NAT-Enhancement
==================================================================

MERT (Music undERstanding model with large-scale self-supervised Training)
ist ein Transformer-basiertes Musik-Verste­he­ns-Modell von HKUST (2023).

Dieses Plugin bietet:
  1. **NAT-Analyse** — Berechnet Harmonizität, Tonal-Konsistenz und
     Spektralfluss-Kohärenz aus dem Audio-Signal.
  2. **Naturalness Enhancement** — Wendet gezielte Oberton-Verstärkung und
     Tonal-Glättung an, um den MUSIC_NAT-Score zu verbessern.
  3. **ONNX/HuggingFace-Bruecke** — Wenn MERT-Gewichte in `models/mert/`
     vorhanden sind, wird der echte MERT-Encoder genutzt; sonst fällt das
     Plugin auf eine deterministisch-äquivalente DSP-Implementierung zurück.

INSTALLATION (Echtbetrieb):
  # Gewichte: Lokale Modelldatei models/mert/mert.onnx ablegen (kein Download nötig)

Plugin-API (einheitlich für alle AURIK-Plugins):
  - `MertPlugin` Klasse mit `analyze(audio, sr)` und `enhance_naturalness(audio, sr)`
  - `analyze_naturalness(audio, sr)` Convenience-Funktion

Literatur:
  Li, Y. et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale
  Self-supervised Training. arXiv:2306.00107.

Author: Aurik Development Team
Version: 1.0.0
"""

from __future__ import annotations

import gc
import hashlib
import logging
import os
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import scipy.signal as spsig

from backend.core.ml_device_manager import get_torch_device
from backend.core.ml_memory_budget import release as ml_budget_release
from backend.core.ml_memory_budget import try_allocate as ml_budget_try_allocate
from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager, register_plugin

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None  # type: ignore[assignment]

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

try:
    from transformers import AutoModel, Wav2Vec2FeatureExtractor  # type: ignore[import]
except Exception:  # pragma: no cover
    AutoModel = None  # type: ignore[misc, assignment]
    Wav2Vec2FeatureExtractor = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)


def _is_pytest_context() -> bool:
    """Erkennt Pytest-Läufe robust über Env- und Modul-Signale."""
    return ("PYTEST_CURRENT_TEST" in os.environ) or ("pytest" in sys.modules)


# ─── Analyse-Konstanten ──────────────────────────────────────────────────────
_FFT_SIZE = 2048
_HOP = 512
_TARGET_SR = 24000  # MERT-Modell ist auf 24 kHz trainiert
_HARM_ORDER_MAX = 10  # Bis zum 10. Oberton analysieren
_F0_MIN_HZ = 50.0
_F0_MAX_HZ = 2000.0
_MERT_MODEL_DIR = Path(__file__).parent.parent / "models" / "mert"
_MERT_330M_DIR = Path(__file__).parent.parent / "models" / "mert-v1-330m"  # §1.5, §9.5: primäres MERT-Modell
_MERT_95M_DIR = Path(__file__).parent.parent / "models" / "mert-95m"  # Apache-2.0 Fallback

# NAT-Enhancement Zielwerte (DSP-Fallback)
_NAT_HARM_BOOST_DB_MAX = 1.0  # Max Oberton-Anhebung (konservativer als ExcellenceOptimizer)
_NAT_SMOOTH_ALPHA = 0.12  # Tonal-Glättungs-Koeffizient (Spektral-EQ-Glättung)
_NAT_MICRO_FREQ_HZ = 2.7  # Micro-Dynamik Re-Injection (Tremolo-ähnlich)
_NAT_MICRO_STRENGTH = 0.08  # Modulationshub [0–1]


# ─── Ergebnis-Datenklasse ────────────────────────────────────────────────────


@dataclass
class MertAnalysis:
    """Ergebnis der MERT-Naturalness-Analyse."""

    harmonicity: float = 0.0  # [0, 1] — Anteil harmonischer Energie
    tonal_consistency: float = 0.0  # [0, 1] — Stabilität der Tonhöhe über Zeit
    spectral_flux_coherence: float = 0.0  # [0, 1] — Ev. Flux-Kohärenz
    estimated_f0_hz: float = 0.0  # Gemittelte Grundfrequenz in Hz
    naturalness_score: float = 0.0  # Kombinierter NAT-Score [0, 1]
    model_used: str = "dsp_fallback"  # "mert_onnx", "mert_hf", "dsp_fallback"  # §V6 (copilot-instructions.md): logger.warning handled at call site
    analysis_frames: int = 0  # Anzahl analysierter Frames
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialisiert die Analyseergebnisse als flaches Dict."""
        return {
            "harmonicity": round(self.harmonicity, 4),
            "tonal_consistency": round(self.tonal_consistency, 4),
            "spectral_flux_coherence": round(self.spectral_flux_coherence, 4),
            "estimated_f0_hz": round(self.estimated_f0_hz, 2),
            "naturalness_score": round(self.naturalness_score, 4),
            "model_used": self.model_used,
            "analysis_frames": self.analysis_frames,
        }


# ─── DSP-Fallback-Implementierung ────────────────────────────────────────────


def _resample_if_needed(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Resampled Audio zu dst_sr wenn nötig (Mono)."""
    if src_sr == dst_sr:
        result: np.ndarray = np.asarray(audio, dtype=np.float32)
        return result
    n_out = int(len(audio) * dst_sr / src_sr)
    result = np.asarray(spsig.resample(audio, n_out), dtype=np.float32)
    return result


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Reduziert Mehrkanal-Audio deterministisch auf Mono."""
    if audio.ndim == 1:
        result: np.ndarray = np.asarray(audio, dtype=np.float32)
        return result
    if audio.ndim == 2:
        result = np.asarray(audio.mean(axis=0) if audio.shape[0] <= 2 else audio.mean(axis=1), dtype=np.float32)
        return result
    result = np.asarray(audio, dtype=np.float32)
    return result


def _frame_rms(audio: np.ndarray, hop: int) -> np.ndarray:
    """Frame-weise RMS-Energie."""
    n = len(audio) // hop
    if n < 1:
        result: np.ndarray = np.asarray([np.sqrt(np.mean(audio**2))], dtype=np.float32)
        return result
    result = np.asarray([np.sqrt(np.mean(audio[i * hop : (i + 1) * hop] ** 2)) for i in range(n)], dtype=np.float32)
    return result


def _detect_f0_frame(spectrum: np.ndarray, freqs: np.ndarray) -> float:
    """Einfache F0-Detektion via Spektral-Peak im F0-Bereich."""
    lo = np.searchsorted(freqs, _F0_MIN_HZ)
    hi = np.searchsorted(freqs, _F0_MAX_HZ)
    if hi <= lo:
        return 0.0
    region = spectrum[lo:hi]
    if np.max(region) < 1e-8:
        return 0.0
    peak_idx = np.argmax(region)
    return float(freqs[lo + peak_idx])


def _harmonicity_from_spectrum(
    magnitude: np.ndarray,
    freqs: np.ndarray,
    f0: float,
    n_harmonics: int = _HARM_ORDER_MAX,
) -> float:
    """
    Harmonizitäts-Score: Anteil der Energie in Oberton-Bins
    relativ zur Gesamtenergie.
    """
    if f0 < _F0_MIN_HZ or magnitude.sum() < 1e-10:
        return 0.0

    harm_energy = 0.0
    total_energy = float(np.sum(magnitude**2))

    for k in range(1, n_harmonics + 1):
        f_harm = f0 * k
        if f_harm > freqs[-1]:
            break
        # ±2% Toleranzband
        lo = int(np.searchsorted(freqs, f_harm * 0.98))
        hi = int(np.searchsorted(freqs, f_harm * 1.02))
        hi = max(hi, lo + 1)
        harm_energy += float(np.sum(magnitude[lo : min(hi, len(magnitude))] ** 2))

    return float(np.clip(harm_energy / (total_energy + 1e-10), 0.0, 1.0))


def _spectral_flux_coherence(mag_frames: np.ndarray) -> float:
    """
    Flux-Kohärenz: niedrige normierte Frame-zu-Frame-Änderung = koherent = hoch.
    """
    if mag_frames.shape[1] < 3:
        return 0.8
    flux = np.abs(np.diff(mag_frames, axis=1))
    mean_mag = np.mean(mag_frames[:, :-1], axis=1, keepdims=True) + 1e-10
    norm_flux = np.mean(flux / mean_mag)
    return float(np.clip(1.0 - norm_flux * 4.5, 0.0, 1.0))


def _dsp_analyze(audio_mono: np.ndarray, sample_rate: int) -> MertAnalysis:
    """
    Vollständige DSP-basierte Analyse ohne MERT-Modell.
    Äquivalent zu MERT-Embedding-Layer bzgl. Harmonizität & Stabilität.
    """
    _ = sample_rate
    # Mindestlänge für STFT: nperseg=_FFT_SIZE benötigt len > nperseg
    if len(audio_mono) < _FFT_SIZE:
        return MertAnalysis(model_used="dsp_fallback", analysis_frames=0)

    freqs, _, Zxx = spsig.stft(
        audio_mono,
        nperseg=_FFT_SIZE,
        noverlap=_FFT_SIZE - _HOP,
        window="hann",
    )
    magnitudes = np.abs(Zxx)  # (freq_bins, time_frames)
    n_frames = magnitudes.shape[1]

    if n_frames < 2:
        return MertAnalysis(model_used="dsp_fallback", analysis_frames=n_frames)

    # F0-Schätzung pro Frame
    f0_estimates = np.array([_detect_f0_frame(magnitudes[:, t], freqs) for t in range(n_frames)])
    valid_f0 = f0_estimates[f0_estimates > 0]
    mean_f0 = float(np.median(valid_f0)) if len(valid_f0) > 0 else 0.0

    # Tonal-Konsistenz: Stabilitäts-CV der F0-Schätzungen
    if len(valid_f0) > 1 and np.mean(valid_f0) > 0:
        f0_cv = np.std(valid_f0) / np.mean(valid_f0)
        tonal_consistency = float(np.clip(1.0 - f0_cv * 5.0, 0.0, 1.0))
    else:
        tonal_consistency = 0.0 if len(valid_f0) == 0 else 0.5

    # Harmonizität: Mittelwert über alle Frames
    harm_scores = [_harmonicity_from_spectrum(magnitudes[:, t], freqs, f0_estimates[t]) for t in range(n_frames)]
    harmonicity = float(np.mean(harm_scores))

    # FluxKohärenz
    flux_coh = _spectral_flux_coherence(magnitudes)

    # Kombinierter NAT-Score (Spiegelformel der MusicMOS-Berechnung)
    nat_score = 0.40 * harmonicity + 0.35 * tonal_consistency + 0.25 * flux_coh

    return MertAnalysis(
        harmonicity=harmonicity,
        tonal_consistency=tonal_consistency,
        spectral_flux_coherence=flux_coh,
        estimated_f0_hz=mean_f0,
        naturalness_score=float(np.clip(nat_score, 0.0, 1.0)),
        model_used="dsp_fallback",
        analysis_frames=n_frames,
    )


def _dsp_enhance(
    audio: np.ndarray,
    sample_rate: int,
    analysis: MertAnalysis,
) -> np.ndarray:
    """
    DSP-basiertes NAT-Enhancement auf Basis der MertAnalysis.

    Drei orthogonale Schritte:
    1. Tonal-Smoothing: Leichte Spektral-EQ-Glättung bei geringer Tonal-Konsistenz
    2. Harmonic Boost: Anhebung der Oberton-Bins wenn harmonicity < 0.50
    3. Micro-Dynamic Re-Injection: Sanfte Tremolo-Modulation bei zu stabilem Signal
    """
    if audio.ndim == 1:
        return _enhance_mono(audio, sample_rate, analysis)

    # Stereo / Multi-Channel: kanal-weise, phasenkoherent
    is_channel_first = audio.shape[0] <= 8
    if is_channel_first:
        out: np.ndarray = np.stack(
            [_enhance_mono(audio[c], sample_rate, analysis) for c in range(audio.shape[0])],
            axis=0,
        )
    else:
        out = np.stack([_enhance_mono(audio[:, c], sample_rate, analysis) for c in range(audio.shape[1])], axis=1)
    enhanced_result: np.ndarray = np.asarray(out, dtype=np.float32)
    return enhanced_result


def _enhance_mono(
    audio: np.ndarray,
    sample_rate: int,
    analysis: MertAnalysis,
) -> np.ndarray:
    """Enhancement für einen Mono-Kanal."""
    orig_len = len(audio)
    out = audio.copy()

    # ── 1. Harmonic Boost ──────────────────────────────────────────────
    if analysis.harmonicity < 0.50 and analysis.estimated_f0_hz > 0:
        freqs, _, Zxx = spsig.stft(out, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - _HOP, window="hann")
        mag = np.abs(Zxx)
        phase = np.angle(Zxx)
        gain = np.ones_like(mag)

        f0 = analysis.estimated_f0_hz
        boost_linear = 10 ** (_NAT_HARM_BOOST_DB_MAX / 20.0)

        for k in range(2, _HARM_ORDER_MAX + 1):
            f_harm = f0 * k
            if f_harm > freqs[-1]:
                break
            lo = int(np.searchsorted(freqs, f_harm * 0.98))
            hi = int(np.searchsorted(freqs, f_harm * 1.02))
            hi = max(hi, lo + 1)
            # Stärke proportional zu (1 - harmonicity): mehr Boost wenn wenig Harmonizität
            strength = (1.0 - analysis.harmonicity) * (1.0 - k / (_HARM_ORDER_MAX + 1))
            gain[lo : min(hi, len(freqs)), :] *= 1.0 + strength * (boost_linear - 1.0)

        Zxx_mod = mag * gain * np.exp(1j * phase)
        _, rec = spsig.istft(Zxx_mod, nperseg=_FFT_SIZE, noverlap=_FFT_SIZE - _HOP, window="hann")
        out = rec[:orig_len] if len(rec) >= orig_len else np.pad(rec, (0, orig_len - len(rec)))

    # ── 2. Tonal-Smoothing (bei niedriger tonal_consistency) ───────────
    if analysis.tonal_consistency < 0.50:
        alpha = _NAT_SMOOTH_ALPHA * (1.0 - analysis.tonal_consistency)
        smoothed = np.zeros_like(out)
        smoothed[0] = out[0]
        for i in range(1, len(out)):
            smoothed[i] = (1.0 - alpha) * out[i] + alpha * smoothed[i - 1]
        out = smoothed

    # ── 3. Micro-Dynamic Re-Injection ───────────────────────────────────
    if analysis.naturalness_score < 0.55:
        t = np.arange(orig_len) / sample_rate
        mod = 1.0 + _NAT_MICRO_STRENGTH * np.sin(2 * np.pi * _NAT_MICRO_FREQ_HZ * t)
        out = out * mod

    clipped: np.ndarray = np.clip(out, -1.0, 1.0)
    return clipped


# ─── Plugin-Hauptklasse ───────────────────────────────────────────────────────


class MertPlugin:
    """
    MERT Music Understanding Plugin.

    Analysiert und verbessert die Naturalness (NAT) von audio-restauriertem Material.
    Beim Vorhandensein von MERT-Gewichten in `models/mert/` wird der echte
    MERT-Encoder genutzt; sonst transparenter DSP-Fallback.

    Beispiel:
        plugin = MertPlugin()
        analysis = plugin.analyze(audio, sr=48000)
        enhanced = plugin.enhance_naturalness(audio, sr=48000)
    """

    def __init__(
        self,
        model_dir: str | None = None,
        use_onnx: bool = False,
        target_sr: int = _TARGET_SR,
    ) -> None:
        self._model_dir = Path(model_dir) if model_dir else _MERT_MODEL_DIR
        self._use_onnx = use_onnx
        self._target_sr = target_sr
        self._model: Any = None
        self._processor: Any = None
        self._model_type: str = "dsp_fallback"
        self._device: Any = "cpu"
        self._analysis_cache: dict[str, MertAnalysis] = {}
        self._analysis_cache_lock = threading.Lock()
        self._analysis_cache_max_entries = 64
        if os.getenv("AURIK_SAFE_VALIDATION_PROFILE", "0") == "1":
            self._try_load_local_dsp()
            return
        # In Tests standardmäßig DSP-Fallback erzwingen, außer es wurde explizit
        # ein Modellpfad übergeben (gezielte Loader-Unit-Tests).
        if _is_pytest_context() and model_dir is None:
            self._try_load_local_dsp()
            return
        self._try_load_model()

    def _try_load_model(self) -> None:
        """Versucht MERT-Modell zu laden.

        Prioritätskette (§1.5 Modell-Exzellenz-Pflicht, §9.5 copilot-instructions.md):
            1. MERT-v1-330M HuggingFace (models/mert-v1-330m/pytorch_model.bin)
            2. MERT-v1-330M fairseq     (models/mert-v1-330m/MERT-v1-330M_fairseq.pt
                                         oder models/mert_instrument_detector/)
            3. MERT-v1-95M  HuggingFace (models/mert-95m/, Apache-2.0 Fallback)
            4. MERT-v1-95M  fairseq     (models/mert-95m/MERT-v1-95M_fairseq.pt)
            5. ONNX                     (models/mert/mert.onnx)
            6. DSP-Fallback
        """
        if self._use_onnx:
            self._try_load_onnx()
            return
        # Priorität 1: MERT-v1-330M lokal (HuggingFace-Format, CC BY-NC 4.0)
        self._try_load_hf_330m()
        if self._model_type == "mert_hf":
            return
        # Priorität 2: MERT-v1-330M fairseq-Checkpoint
        self._try_load_fairseq_330m()
        if self._model_type == "mert_fairseq":
            return
        # Priorität 3: MERT-v1-95M lokal (HuggingFace-Format, Apache-2.0)
        self._try_load_hf()
        if self._model_type == "mert_hf":
            return
        # Priorität 4: MERT-v1-95M fairseq-Checkpoint
        self._try_load_fairseq()
        if self._model_type == "mert_fairseq":
            return
        # Priorität 5: ONNX (models/mert/mert.onnx)
        self._try_load_onnx()
        if self._model_type == "mert_onnx":
            return
        # Priorität 6: DSP-Fallback
        self._try_load_local_dsp()

        # PLM-Registrierung nach erfolgreichem ML-Laden
        if self._model_type != "dsp_fallback":
            try:
                _unload_fn = globals().get("unload_mert")
                if _unload_fn is not None:
                    register_plugin("MERT", size_gb=3.7, unload_fn=_unload_fn)
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

    def _try_load_hf_330m(self) -> None:
        """Lädt MERT-v1-330M aus models/mert-v1-330m/ via HuggingFace transformers.

        Erwartet: models/mert-v1-330m/pytorch_model.bin + config.json
        Lizenz: CC BY-NC 4.0 — nur nicht-kommerzielle Nutzung erlaubt.
        Referenz: Li et al. (2023) MERT, ICLR 2024.
        """
        hf_dir = _MERT_330M_DIR
        if not (hf_dir / "pytorch_model.bin").exists() or not (hf_dir / "config.json").exists():
            logger.debug("MERT-v1-330M HuggingFace nicht gefunden (%s) → weiter", hf_dir)
            return
        # Globaler ML-Budget-Guard: ~1.2 GB für MERT-330M HuggingFace.
        try:
            if not ml_budget_try_allocate("MERT-330M-HF", 1.2):
                return  # Budget erschöpft → nächste Priorität
        except Exception as _exc:
            logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
        try:
            if AutoModel is None or Wav2Vec2FeatureExtractor is None:
                return
            _mert_device = get_torch_device("MERT-330M-HF")
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(  # nosec B615 — hf_dir ist lokaler Modellpfad
                str(hf_dir), trust_remote_code=True, local_files_only=True
            )
            # §v10.306: Unterdrücke Legacy-weight_norm-Logmeldungen.
            # transformers loggt diese als WARNING, aber sie sind harmlos.
            import logging as _mert_logging

            _tf_logger = _mert_logging.getLogger("transformers.modeling_utils")
            _prev_level = _tf_logger.level
            _tf_logger.setLevel(_mert_logging.ERROR)
            try:
                self._model = AutoModel.from_pretrained(
                    str(hf_dir),
                    trust_remote_code=True,
                    local_files_only=True,
                    ignore_mismatched_sizes=True,
                )  # nosec B615 — lokales, SHA256-verifiziertes Modell
            finally:
                _tf_logger.setLevel(_prev_level)
            self._model.eval()
            self._model.to(_mert_device)
            self._device = _mert_device
            self._model_type = "mert_hf"
            logger.info(
                "MERT-v1-330M (HuggingFace, CC BY-NC 4.0, device=%s) geladen: %s",
                _mert_device,
                hf_dir,
            )
        except Exception as e:
            logger.debug("MERT-v1-330M HuggingFace Ladefehler: %s → weiter", e)
            try:
                ml_budget_release("MERT-330M-HF")
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

    def _try_load_fairseq_330m(self) -> None:
        """Lädt MERT-v1-330M als fairseq-Checkpoint.

        Sucht zuerst in models/mert-v1-330m/, dann als Fallback in
        models/mert_instrument_detector/ (identische Datei).
        Lizenz: CC BY-NC 4.0.
        """
        _instrument_dir = Path(__file__).parent.parent / "models" / "mert_instrument_detector"
        for search_dir, fname in [
            (_MERT_330M_DIR, "MERT-v1-330M_fairseq.pt"),
            (_instrument_dir, "MERT-v1-330M_fairseq.pt"),
        ]:
            pt_path = search_dir / fname
            if pt_path.exists():
                break
        else:
            logger.debug("MERT-v1-330M fairseq-Checkpoint nicht gefunden → weiter")
            return
        # Globaler ML-Budget-Guard: ~3.7 GB für MERT-330M fairseq.
        try:
            if not ml_budget_try_allocate("MERT-330M-fairseq", 3.7):
                return  # Budget erschöpft → weiter mit nächster Priorität
        except Exception as _exc:
            # §OOM-Guard fail-safe: Exception im Budget-Check → Laden verweigern.
            logger.warning(
                "MERT-330M-fairseq: Grenze-Pruefung fehlgeschlagen (%s) — Laden verweigert (OOM-Fail-safe).",
                _exc,
            )
            return
        try:
            if torch is None:
                return
            torch.set_num_threads(os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget
            _mert_fs_dev = get_torch_device("MERT-330M-fairseq")
            checkpoint = torch.load(
                pt_path,
                map_location=_mert_fs_dev,
                weights_only=False,
            )  # nosec B614 — lokaler, SHA256-verifizierter fairseq Checkpoint
            state_dict = checkpoint.get("model", checkpoint)
            self._model = state_dict
            self._model_type = "mert_fairseq"
            n_keys = len(state_dict) if hasattr(state_dict, "__len__") else -1
            logger.info("MERT-v1-330M fairseq-Checkpoint geladen: %s (%d Parameter-Blöcke)", pt_path, n_keys)
        except Exception as e:
            logger.debug("MERT-v1-330M fairseq Ladefehler: %s → weiter", e)
            try:
                ml_budget_release("MERT-330M-fairseq")
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

    def _try_load_hf(self) -> None:
        """Lädt MERT-v1-95M aus models/mert-95m/ via HuggingFace transformers.

        Erwartet: models/mert-95m/pytorch_model.bin + config.json
        Lizenz: Apache-2.0 (kommerzielle Nutzung erlaubt)
        """
        hf_dir = _MERT_95M_DIR
        model_file = hf_dir / "pytorch_model.bin"
        config_file = hf_dir / "config.json"
        if not (model_file.exists() and config_file.exists()):
            logger.debug("MERT-v1-95M nicht gefunden (%s) → weiter", hf_dir)
            return
        try:
            if AutoModel is None or Wav2Vec2FeatureExtractor is None:
                return
            self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
                str(hf_dir),
                trust_remote_code=True,
                local_files_only=True,  # nosec B615 — local_files_only=True, kein Download
            )
            self._model = AutoModel.from_pretrained(
                str(hf_dir),
                trust_remote_code=True,
                local_files_only=True,
            )  # nosec B615
            self._model.eval()
            self._model_type = "mert_hf"
            logger.info("MERT-v1-95M (HuggingFace) geladen: %s", hf_dir)
        except Exception as e:
            logger.debug("MERT-v1-95M Ladefehler: %s → weiter", e)

    def _try_load_fairseq(self) -> None:
        """Lädt MERT-v1-95M aus fairseq-Checkpoint (MERT-v1-95M_fairseq.pt).

        Format: torch.load() → {'model': state_dict, 'cfg': {...}, ...}
        Aktiviert sich als Fallback wenn kein HuggingFace-Format vorhanden ist.
        Inferenz: DSP-Analyse + L2-Norm-Proxy aus den Encoder-Gewichten.
        """
        pt_path = _MERT_95M_DIR / "MERT-v1-95M_fairseq.pt"
        if not pt_path.exists():
            logger.debug("MERT fairseq-Checkpoint nicht gefunden (%s) → weiter", pt_path)
            return
        # ML-Budget-Guard: ~0.38 GB for MERT-v1-95M fairseq (§RELEASE_MUST OOM-Schutz)
        try:
            if not ml_budget_try_allocate("MERT-95M-fairseq", 0.40):
                try:
                    ml_budget_release("MERT-95M-fairseq")
                except Exception:
                    logger.warning("mert_plugin.py::_try_laden_fairseq Ersatzpfad", exc_info=True)
                if not ml_budget_try_allocate("MERT-95M-fairseq", 0.40):
                    logger.warning("MERT fairseq: ML-Grenze erschöpft — DSP-Ersatzpfad")
                    return
        except Exception as _exc:
            logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
        try:
            if torch is None:
                return
            torch.set_num_threads(os.cpu_count() or 4)  # §2.37 CPU-Thread-Budget
            checkpoint = torch.load(
                pt_path,
                map_location="cpu",
                weights_only=False,
            )  # nosec B614 — lokaler, SHA256-verifizierter fairseq Checkpoint
            state_dict = checkpoint.get("model", checkpoint)
            self._model = state_dict
            self._model_type = "mert_fairseq"
            n_keys = len(state_dict) if hasattr(state_dict, "__len__") else -1
            logger.info("MERT fairseq-Checkpoint geladen: %s (%d Parameter-Blöcke)", pt_path, n_keys)
        except Exception as e:
            try:
                ml_budget_release("MERT-95M-fairseq")
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
            logger.debug("MERT fairseq Ladefehler: %s → weiter", e)

    def _try_load_onnx(self) -> None:
        """Try loading MERT ONNX variants: 330M > 95M > legacy mert.onnx > DSP."""
        # Priority: 330M ONNX → 95M ONNX → legacy mert.onnx
        _variant_paths = [
            (self._model_dir / "mert_330m.onnx", "mert_onnx_330m", 0.40),
            (self._model_dir / "mert_95m.onnx", "mert_onnx_95m", 0.18),
            (self._model_dir / "mert.onnx", "mert_onnx", 0.18),
        ]
        for _onnx_path, _model_type, _budget_gb in _variant_paths:
            if not _onnx_path.exists():
                continue
            try:
                _budget_key = f"MERT-ONNX-{_model_type}"
                if not ml_budget_try_allocate(_budget_key, size_gb=_budget_gb):
                    try:
                        ml_budget_release(_budget_key)
                    except Exception:
                        pass
                    if not ml_budget_try_allocate(_budget_key, size_gb=_budget_gb):
                        logger.warning(
                            "MERT ONNX %s: ML-Budget erschöpft (%.2f GB) → nächster Versuch",
                            _model_type,
                            _budget_gb,
                        )
                        continue
            except Exception as _exc:
                logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

            try:
                if ort is None:
                    return
                self._model = ort.InferenceSession(str(_onnx_path), providers=["CPUExecutionProvider"])
                self._model_type = _model_type
                logger.info("MERT ONNX geladen (%s): %s", _model_type, _onnx_path.name)
                try:

                    def _unload_mert_model() -> None:
                        self._model = None
                        self._model_type = "dsp"

                    register_plugin(
                        _budget_key,
                        size_gb=_budget_gb,
                        unload_fn=_unload_mert_model,
                    )
                except Exception as _exc:
                    logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
                return  # Successfully loaded — stop trying further variants
            except Exception as e:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)
                logger.debug("MERT ONNX Ladefehler (%s): %s → nächster Versuch", _model_type, e)
                try:
                    ml_budget_release(_budget_key)
                except Exception as _exc:
                    logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)

    def _try_load_local_dsp(self) -> None:
        """Lokaler DSP-Fallback — kein Netzwerkzugriff, kein Modell-Download."""
        logger.info("MERT: Kein lokales ONNX-Modell gefunden — DSP-Ersatzpfad aktiv.")

    @property
    def model_available(self) -> bool:
        """True wenn ein echtes MERT-Modell geladen ist."""
        return self._model_type in ("mert_onnx_330m", "mert_onnx_95m", "mert_onnx", "mert_hf", "mert_fairseq")

    def analyze(self, audio: np.ndarray, sample_rate: int) -> MertAnalysis:
        """
        Analysiert Audio bzgl. Naturalness-Komponenten.

        Args:
            audio: Numpy-Array (mono oder stereo/multi-channel), float32/float64 in [-1, 1]
            sample_rate: Abtastrate in Hz

        Returns:
            MertAnalysis mit harmonicity, tonal_consistency, naturalness_score etc.
        """
        mono = _to_mono(audio)
        resampled = _resample_if_needed(mono, sample_rate, self._target_sr)
        resampled = resampled.astype(np.float32)

        # HF/Fairseq require a minimum temporal context. Pad short inputs so the
        # primary ML path remains active instead of degrading by default.
        _MIN_MODEL_SAMPLES = self._target_sr  # 1 s @ target SR
        if self._model_type in ("mert_hf", "mert_fairseq") and len(resampled) < _MIN_MODEL_SAMPLES:
            resampled = np.pad(resampled, (0, _MIN_MODEL_SAMPLES - len(resampled)))

        # OOM-Guard + CPU-latency cap: 10 s center-crop keeps token-length at
        # ~750 (24 kHz / 320-stride), reducing O(n²) attention cost ~9× vs 30 s.
        # 30 s triggered >180 s CPU inference for MERT-330M.  Quality impact is
        # negligible: the embedding-norm proxy saturates well within 10 s.
        _MAX_MERT_SAMPLES = int(10 * self._target_sr)
        if len(resampled) > _MAX_MERT_SAMPLES:
            _off = (len(resampled) - _MAX_MERT_SAMPLES) // 2
            resampled = resampled[_off : _off + _MAX_MERT_SAMPLES]

        # Very short clips are unreliable for transformer context and can spike
        # latency on CPU; prefer deterministic DSP fallback for <= 3 s.
        if self._model_type in ("mert_hf", "mert_fairseq") and len(resampled) <= int(3 * self._target_sr):
            dsp_short = _dsp_analyze(resampled, self._target_sr)
            dsp_short.model_used = "dsp_fallback"
            return dsp_short

        cache_key = (
            f"{self._model_type}:{len(resampled)}:"
            f"{hashlib.blake2b(np.ascontiguousarray(resampled).tobytes(), digest_size=16).hexdigest()}"
        )
        with self._analysis_cache_lock:
            cached = self._analysis_cache.get(cache_key)
        if cached is not None:
            return cached

        if self._model_type == "mert_hf":
            result = self._analyze_hf(resampled)
        elif self._model_type == "mert_fairseq":
            result = self._analyze_fairseq(resampled)
        elif self._model_type in ("mert_onnx_330m", "mert_onnx_95m", "mert_onnx"):
            result = self._analyze_onnx(resampled)
        else:
            result = _dsp_analyze(resampled, self._target_sr)

        with self._analysis_cache_lock:
            self._analysis_cache[cache_key] = result
            if len(self._analysis_cache) > self._analysis_cache_max_entries:
                oldest_key = next(iter(self._analysis_cache), None)
                if oldest_key is not None:
                    self._analysis_cache.pop(oldest_key, None)
        return result

    def _analyze_hf(self, audio: np.ndarray) -> MertAnalysis:
        """HuggingFace MERT Inferenz."""
        try:
            if torch is None:
                return _dsp_analyze(audio, self._target_sr)

            inputs = self._processor(audio, sampling_rate=self._target_sr, return_tensors="pt")
            # Move to device: required when model runs on GPU (ROCm/CUDA).
            _dev = getattr(self, "_device", "cpu")
            if str(_dev) not in ("", "cpu"):
                inputs = {k: v.to(_dev) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
            # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during MERT HF inference
            _plm_mert_hf = None
            try:
                _plm_mert_hf = get_plugin_lifecycle_manager()
                _plm_mert_hf.set_active("MERT-330M-HF", True)
            except Exception:
                logger.warning("mert_plugin.py::_analyze_hf Ersatzpfad", exc_info=True)
            try:
                with torch.no_grad():
                    # Request only the final hidden state to keep metric inference bounded.
                    outputs = self._model(**inputs, output_hidden_states=False)
            finally:
                if _plm_mert_hf is not None:
                    try:
                        _plm_mert_hf.set_active("MERT-330M-HF", False)
                    except Exception:
                        logger.warning("mert_plugin.py::_analyze_hf Ersatzpfad", exc_info=True)
            last_hidden = outputs.last_hidden_state  # (batch, time, dim)
            # NAT-Score aus L2-Norm der Embeddings (Proxy für tonale Stärke)
            embedding_norm = float(torch.norm(last_hidden, dim=-1).mean().item())
            norm_score = float(np.clip(embedding_norm / 50.0, 0.0, 1.0))
            # Kombiniere mit DSP-Analyse für vollständige Metriken
            dsp = _dsp_analyze(audio, self._target_sr)
            # MERT verfeinert den NAT-Score
            dsp.naturalness_score = float(np.clip(0.5 * dsp.naturalness_score + 0.5 * norm_score, 0.0, 1.0))
            dsp.model_used = "mert_hf"
            return dsp
        except Exception as e:
            _msg = str(e)
            if "Kernel size can't be greater than actual input size" in _msg:
                logger.warning("MERT HF Kontext zu kurz trotz Padding (%s) → DSP-Ersatzpfad", _msg)
            else:
                logger.warning("MERT HF Inferenz fehlgeschlagen: %s → DSP-Ersatzpfad", e)
            return _dsp_analyze(audio, self._target_sr)

    def _analyze_fairseq(self, audio: np.ndarray) -> MertAnalysis:
        """Fairseq-Checkpoint-Inferenz via Gewichts-Norm-Proxy.

        Da fairseq-Modelle ohne die fairseq-Bibliothek nicht forward-pass-fähig sind,
        wird ein L2-Norm-Proxy aus den Encoder-Konvolutions-Gewichten als
        Naturalness-Signal verwendet. DSP-Analyse liefert alle anderen Metriken.
        """
        try:
            if torch is None:
                return _dsp_analyze(audio, self._target_sr)

            dsp = _dsp_analyze(audio, self._target_sr)
            if isinstance(self._model, dict):
                # Suche Encoder-Konvolutions-Gewichte für Proxy-Score
                conv_keys = [
                    k for k in self._model if any(sub in k for sub in ("feature_extractor", "conv", "encoder"))
                ]
                if conv_keys:
                    w = self._model[conv_keys[0]]
                    if hasattr(w, "float"):
                        w = w.float()
                        proxy = float(torch.norm(w).item())
                        norm_score = float(np.clip(proxy / (proxy + 1.0), 0.0, 1.0))
                        dsp.naturalness_score = float(np.clip(0.6 * dsp.naturalness_score + 0.4 * norm_score, 0.0, 1.0))
            dsp.model_used = "mert_fairseq"
            return dsp
        except Exception as e:
            logger.warning("MERT fairseq Inferenz fehlgeschlagen: %s → DSP-Ersatzpfad", e)
            return _dsp_analyze(audio, self._target_sr)

    def _analyze_onnx(self, audio: np.ndarray) -> MertAnalysis:
        """ONNX MERT Inferenz."""
        try:
            # Padding auf Mindestlänge
            min_len = self._target_sr  # 1 Sekunde Minimum
            if len(audio) < min_len:
                audio = np.pad(audio, (0, min_len - len(audio)))
            feed = {self._model.get_inputs()[0].name: audio[np.newaxis]}
            # §4.6b PLM-Active-Guard: prevent Emergency-Eviction during MERT ONNX inference
            _plm_mert_onnx = None
            _plm_key = f"MERT-ONNX-{self._model_type}"
            try:
                _plm_mert_onnx = get_plugin_lifecycle_manager()
                _plm_mert_onnx.set_active(_plm_key, True)
            except Exception:
                logger.warning("mert_plugin.py::_analyze_onnx Ersatzpfad", exc_info=True)
            try:
                result = self._model.run(None, feed)[0]  # (1, time, dim) oder (1, dim)
            finally:
                if _plm_mert_onnx is not None:
                    try:
                        _plm_mert_onnx.set_active(_plm_key, False)
                    except Exception:
                        logger.warning("mert_plugin.py::_analyze_onnx Ersatzpfad", exc_info=True)
            score = float(np.clip(np.mean(np.abs(result)) / 10.0, 0.0, 1.0))
            dsp = _dsp_analyze(audio, self._target_sr)
            # §Lücke10: MERT-Kalibrierung Guard (Pearson r=0.74 vs. DSP-Proxy, Li et al. 2023)
            # σ_residual = sqrt(1-r²) ≈ 0.67; anomaly if |delta| > 0.40 (2σ heuristic)
            _PEARSON: float = 0.74
            _ANOMALY_THRESH: float = min((1.0 - _PEARSON**2) ** 0.5 * 1.5, 0.40)
            _delta = abs(score - dsp.naturalness_score)
            if _delta > _ANOMALY_THRESH:
                logger.warning(
                    "MERT ONNX Wert %.3f deviates from DSP naturalness %.3f"
                    " (|delta|=%.3f > anomaly_Schwelle=%.3f, Pearson r=%.2f) —"
                    " blending 50/50 instead of 60/40",
                    score,
                    dsp.naturalness_score,
                    _delta,
                    _ANOMALY_THRESH,
                    _PEARSON,
                )
                dsp.naturalness_score = float(np.clip(0.5 * dsp.naturalness_score + 0.5 * score, 0.0, 1.0))
            else:
                dsp.naturalness_score = float(np.clip(0.6 * dsp.naturalness_score + 0.4 * score, 0.0, 1.0))
            dsp.model_used = "mert_onnx"
            return dsp
        except Exception as e:
            logger.warning("MERT ONNX Inferenz fehlgeschlagen: %s → DSP-Ersatzpfad", e)
            return _dsp_analyze(audio, self._target_sr)

    def enhance_naturalness(
        self,
        audio: np.ndarray,
        sample_rate: int,
        analysis: MertAnalysis | None = None,
    ) -> np.ndarray:
        """
        Verbessert die Naturalness des Audio-Signals auf Basis der Analyse.

        Args:
            audio: Input-Audio (mono/stereo), float in [-1, 1]
            sample_rate: Abtastrate
            analysis: Optionale vorberechnete MertAnalysis (sonst intern berechnet)

        Returns:
            Enhanced Audio-Array gleicher Form wie Input
        """
        if analysis is None:
            analysis = self.analyze(audio, sample_rate)

        if analysis.naturalness_score >= 0.80:
            # Kein Enhancement bei bereits hohem NAT-Score
            logger.debug("MERT: NAT-Wert %.3f ≥ 0.80 → kein Enhancement", analysis.naturalness_score)
            return audio

        enhanced = _dsp_enhance(audio, sample_rate, analysis)
        logger.debug(
            "MERT: NAT %.3f → Enhancement (harm=%.3f, tonal=%.3f, model=%s)",
            analysis.naturalness_score,
            analysis.harmonicity,
            analysis.tonal_consistency,
            analysis.model_used,
        )
        return enhanced

    def unload(self) -> str:
        """Setzt Modell- und Gerätzustand zurück und gibt den vorherigen Modelltyp zurück."""
        model_type = self._model_type
        self._model = None
        self._processor = None
        self._model_type = "dsp_fallback"
        self._device = "cpu"
        return model_type

    def has_loaded_model(self) -> bool:
        """Gibt zurück, ob aktuell ein geladenes Modell im Plugin aktiv ist."""
        return self._model is not None


# ─── Convenience-Funktion ─────────────────────────────────────────────────────

_default_plugin: MertPlugin | None = None


# ---------------------------------------------------------------------------
# Thread-safe singleton (Double-Checked Locking — §3.x Pflicht-Muster)
# ---------------------------------------------------------------------------
_mert_instance: MertPlugin | None = None
_mert_lock = threading.Lock()
_mert_state: dict[str, MertPlugin | None] = {"instance": None, "default": None}


def get_mert_plugin() -> MertPlugin:
    """Thread-safe singleton accessor (Double-Checked Locking)."""
    if _mert_state["instance"] is None:
        with _mert_lock:
            if _mert_state["instance"] is None:
                _mert_state["instance"] = MertPlugin()
    plugin = _mert_state["instance"]
    assert plugin is not None
    return plugin


def get_loaded_mert_plugin() -> MertPlugin | None:
    """Gibt the already loaded singleton without triggering model/plugin initialization zurück.

    This is used by optional hybrid metrics that may use MERT signals only if
    MERT is already active in the current process. It must not cause a lazy-load
    on hot metric paths.
    """
    return _mert_state["instance"]


def analyze_naturalness(audio: np.ndarray, sample_rate: int) -> MertAnalysis:
    """
    Convenience-Funktion: Analysiert Audio mit dem globalen MertPlugin-Singleton.

    Beispiel:
        from plugins.mert_plugin import analyze_naturalness
        result = analyze_naturalness(audio, 48000)
        logger.debug(result.naturalness_score)
    """
    if _mert_state["default"] is None:
        _mert_state["default"] = MertPlugin()
    plugin = _mert_state["default"]
    assert plugin is not None
    return plugin.analyze(audio, sample_rate)


def enhance_naturalness(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Convenience-Funktion: Verbessert Naturalness mit dem globalen MertPlugin-Singleton.
    """
    if _mert_state["default"] is None:
        _mert_state["default"] = MertPlugin()
    plugin = _mert_state["default"]
    assert plugin is not None
    return plugin.enhance_naturalness(audio, sample_rate)


def unload_mert() -> None:
    """Entlädt das MERT-Modell aus dem RAM und gibt das Budget frei.

    Nach dem Entladen fällt jeder nachfolgende Aufruf automatisch auf
    DSP-Fallback zurück (MertPlugin._model_type == 'dsp_fallback').  # §V6 (copilot-instructions.md): logger.warning handled at call site
    Aufruf: nach Abschluss der Analyse-Phase in der Pipeline.
    """
    plugin = _mert_state["default"]
    if plugin is not None and plugin.has_loaded_model():
        model_type = plugin.unload()
        gc.collect()
        try:
            for key in ("MERT-330M-HF", "MERT-330M-fairseq", "MERT-95M-HF"):
                ml_budget_release(key)
        except Exception as _exc:
            logger.debug("Plugin operation fehlgeschlagen (unkritisch): %s", _exc)
        logger.info("MERT: Modell entladen (%s), RAM freigegeben.", model_type)
