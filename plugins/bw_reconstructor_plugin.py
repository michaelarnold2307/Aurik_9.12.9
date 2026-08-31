#!/usr/bin/env python3
"""
BW Reconstructor Plugin für Aurik — ONNX-basierte Bandbreiten-Rekonstruktion.

Rekonstruiert fehlende hohe Frequenzen in bandbreitenbegrenztem Audiomaterial
(z. B. Telefonaufnahmen, alte Kassetten, 8-kHz-Streams) durch ein trainiertes
U-Net, das im Mel-Spektrogramm-Raum arbeitet.

ONNX-Modell: ~50 MB, CPU-only, keine GPU nötig.
Voraussetzung: pip install onnxruntime soundfile scipy

Integration in die Aurik-Pipeline:
    from plugins.bw_reconstructor_plugin import BWReconstructorPlugin
    plugin = BWReconstructorPlugin()
    reconstructed = plugin.reconstruct(audio, sr, cutoff_hz=4000)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np

logger = logging.getLogger(__name__)

# ── Verfügbarkeits-Checks ──────────────────────────────────────────────────

try:
    import onnxruntime as ort

    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False
    logger.debug("onnxruntime nicht installiert — BW Reconstructor nicht verfügbar")

try:
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# ── Default-Pfade ──────────────────────────────────────────────────────────

_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "bw_reconstructor"
_DEFAULT_ONNX_PATH = _DEFAULT_MODEL_DIR / "bw_reconstructor.onnx"

# Modulebene (§G174): Flags ohne Backend-Abhängigkeiten.
try:
    from backend.core.music_model_flags import resolve_model_path as _resolve_model_path_flag

    _FLAGS_AVAILABLE = True
except Exception as _flags_exc:
    _FLAGS_AVAILABLE = False
    _resolve_model_path_flag = None  # type: ignore[assignment]
    logger.debug("BWReconstructor: music_model_flags nicht ladbar: %s", _flags_exc)


def _resolve_default_path() -> Path:
    """§v10.19 Feature-Flag-Routing: BW-v5 (selbst trainiert) vor v1 (Legacy).

    use_bw_v5=True und v5-ONNX vorhanden → bestes trainiertes Modell aktiv.
    Sonst v1 — identisches U-Net-Interface, kein Risiko.
    """
    if _FLAGS_AVAILABLE and _resolve_model_path_flag is not None:
        _p = _resolve_model_path_flag("bw")
        if _p is not None and Path(_p).exists():
            return Path(_p)
    return _DEFAULT_ONNX_PATH


class BWReconstructorPlugin:
    """Bandwidth Reconstructor: U-Net im Mel-Spektrogramm-Raum via ONNX.

    Konventionen (siehe train_bw_reconstructor.py):
      - Samplerate: 22050 Hz
      - n_fft=2048, hop_length=512, n_mels=256
      - 256×256 Mel-Spektrogramm → entspricht ~6 s Audio
      - Input: bandbreitenbegrenztes Mel
      - Output: rekonstruiertes Mel (Residual additiv auf Input)
    """

    _BUDGET_NAME: str = "BWReconstructor"
    _BUDGET_SIZE_GB: float = 0.10  # ~50 MB ONNX + Ort-Session

    # Spektrogramm-Parameter (müssen mit Training übereinstimmen)
    _N_FFT: int = 2048
    _HOP_LENGTH: int = 512
    _N_MELS: int = 256
    _SR: int = 22050
    _N_FRAMES: int = 256  # Ziel-Frames pro Segment

    def __init__(self, model_path: str | None = None):
        self._session: ort.InferenceSession | None = None
        self._model_path: Path | None = None
        self._is_v5: bool = False  # v5 = Waveform-Domäne (waveform+cutoff Inputs)

        if not _ONNX_AVAILABLE:
            logger.warning("BWReconstructorPlugin: onnxruntime fehlt — Plugin inaktiv.")
            return

        path = Path(model_path) if model_path else _resolve_default_path()
        if not path.exists():
            logger.warning(
                "BWReconstructorPlugin: ONNX-Modell nicht gefunden unter %s. "
                "Bitte train_bw_reconstructor.py und Ausgabe_bw_to_onnx.py ausführen, "
                "oder --model-path angeben.",
                path,
            )
            return

        try:
            # ML-Memory-Budget prüfen
            try:
                from backend.core.ml_memory_budget import try_allocate

                if not try_allocate(self._BUDGET_NAME, size_gb=self._BUDGET_SIZE_GB):
                    logger.info("BWReconstructor: ML-Grenze erschöpft — Plugin inaktiv.")
                    return
            except ImportError:
                pass

            # ONNX-Session mit CPU-Provider
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
            self._model_path = path

            # §v10.19 Interface-Detektion: v5 ist ein Waveform-U-Net
            # (Inputs waveform+cutoff), v1–v4 sind Mel-Domäne (Input input).
            try:
                _in_names = [i.name for i in self._session.get_inputs()]
                self._is_v5 = _in_names == ["waveform", "cutoff"]
                if self._is_v5:
                    logger.info("BWReconstructor: v5-Waveform-Interface erkannt (FiLM-Cutoff-Conditioning)")
            except Exception:
                self._is_v5 = False

            # Plugin-Lifecycle registrieren
            try:
                from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                _self = self
                _reg_plm(
                    self._BUDGET_NAME,
                    size_gb=self._BUDGET_SIZE_GB,
                    unload_fn=lambda s=_self: setattr(s, "_session", None),  # type: ignore[misc]
                )
            except Exception:
                logger.debug("PLM-Registrierung nicht möglich (unkritisch)")

            logger.info("BWReconstructorPlugin geladen: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)

        except Exception as exc:
            logger.error("BWReconstructorPlugin: Fehler beim Laden: %s", exc)
            self._session = None

    @property
    def available(self) -> bool:
        """True wenn das ONNX-Modell geladen und bereit ist."""
        return self._session is not None and _ONNX_AVAILABLE

    # ── Kernfunktion: Rekonstruktion ───────────────────────────────────────

    def reconstruct(
        self,
        audio: np.ndarray,
        sr: int,
        cutoff_hz: float | None = None,
        blend_strength: float = 1.0,
    ) -> np.ndarray:
        """Rekonstruiert hohe Frequenzen im Audiosignal.

        Args:
            audio: Eingabe-Audio (1D Mono oder 2D Stereo, float32/float64 im Bereich [-1,1]).
            sr: Original-Samplerate (wird intern auf 22050 Hz resampelt).
            cutoff_hz: Bekannte Bandbreitengrenze (optional; sonst automatisch geschätzt).
            blend_strength: 0.0 = Original, 1.0 = volle Rekonstruktion.

        Returns:
            Rekonstruiertes Audiosignal in der originalen Samplerate.
        """
        if not self.available:
            logger.warning("BWReconstructorPlugin nicht verfügbar — gebe Originalsignal zurück.")
            return audio.copy()

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        # Stereo → Mono für die Verarbeitung
        is_stereo = audio.ndim == 2 and audio.shape[0] == 2
        if is_stereo:
            audio_mono = audio.mean(axis=0)
        else:
            audio_mono = audio.squeeze()
            if audio_mono.ndim > 1:
                audio_mono = audio_mono.mean(axis=0)

        # Resampling auf 22050 Hz
        audio_22k = self._resample(audio_mono.astype(np.float64), sr, self._SR)

        # Automatische Cutoff-Schätzung
        if cutoff_hz is None:
            cutoff_hz = self._estimate_cutoff(audio_22k, self._SR)
            logger.debug("BWReconstructor: geschätzte Bandbreitengrenze = %.0f Hz", cutoff_hz)

        # §v10.19 v5-Zweig: Waveform-Domäne (train_bw_v5.py-Konventionen)
        if self._is_v5:
            audio_reconstructed = self._reconstruct_v5_waveform(audio_22k, cutoff_hz, blend_strength)
        else:
            # Mel-Spektrogramm
            mel_input = self._audio_to_mel(audio_22k)

            # Verarbeitung in überlappenden Segmenten
            mel_reconstructed = self._process_segments(mel_input)

            # Blend: Original <-> Rekonstruktion
            if blend_strength < 1.0:
                mel_reconstructed = blend_strength * mel_reconstructed + (1.0 - blend_strength) * mel_input

            # Zurück zu Audio
            audio_reconstructed = self._mel_to_audio(mel_reconstructed)

        # Resampling zurück zur Original-Samplerate
        if sr != self._SR:
            audio_reconstructed = self._resample(audio_reconstructed, self._SR, sr)

        # Länge anpassen
        if len(audio_reconstructed) > len(audio_mono):
            audio_reconstructed = audio_reconstructed[: len(audio_mono)]
        elif len(audio_reconstructed) < len(audio_mono):
            audio_reconstructed = np.pad(audio_reconstructed, (0, len(audio_mono) - len(audio_reconstructed)))

        # Stereo: linken und rechten Kanal separat rekonstruieren
        if is_stereo:
            gain = np.clip(
                audio_reconstructed / (np.abs(audio_mono[: len(audio_reconstructed)]).max() + 1e-8),
                -2.0,
                2.0,
            )
            audio_out = np.zeros_like(audio)
            for ch in range(2):
                audio_out[ch] = audio[ch, : len(gain)] * gain
        else:
            audio_out = audio_reconstructed

        return cast(np.ndarray, (np.nan_to_num(audio_out, nan=0.0, posinf=0.0, neginf=0.0)))

    # ── v5-Waveform-Pfad (§v10.19, train_bw_v5.py-Konventionen) ─────────────

    _V5_CHUNK: int = 66160  # 3 s @ 22050, multiple von 16 (4 Downsampling-Stufen)
    _V5_HOP: int = 33080  # 50 % Überlapp für Crossfade

    def _reconstruct_v5_waveform(
        self,
        audio_22k: np.ndarray,
        cutoff_hz: float,
        blend_strength: float,
    ) -> np.ndarray:
        """Waveform-Domäne Inferenz: bandbegrenztes Audio → Vollband-Waveform.

        Trainingskonventionen (train_bw_v5.py):
          - SR 22050, Chunk 3 s (66160 Samples, multiple von 16)
          - Input: Butterworth-lowpass-bandbegrenzt bei cutoff_hz
          - Condition: cutoff_hz / 11025.0 (FiLM)
          - Output: rekonstruierte Vollband-Waveform (L1 + MR-STFT trainiert)
        """
        if self._session is None:
            return audio_22k.copy()

        n = len(audio_22k)
        # Cutoff normalisieren + auf Trainingsbereich klammern (3000–11000 Hz)
        c_norm = float(np.clip(cutoff_hz / 11025.0, 3000.0 / 11025.0, 11000.0 / 11025.0))

        from scipy.signal import butter, sosfilt

        _sos = butter(6, max(cutoff_hz, 3000.0) / (self._SR / 2.0), btype="low", output="sos")

        out_accum = np.zeros(n, dtype=np.float64)
        out_weight = np.zeros(n, dtype=np.float64)
        pos = 0
        ramp = np.linspace(0.0, 1.0, self._V5_HOP, endpoint=False)  # lineare Crossfade-Rampe

        while pos < n:
            chunk = audio_22k[pos : pos + self._V5_CHUNK]
            if len(chunk) < self._V5_CHUNK:
                chunk = np.pad(chunk, (0, self._V5_CHUNK - len(chunk)), mode="reflect")

            # Training-identische Bandbegrenzung (y_lim = butter_lp(y, cutoff))
            x_lim = sosfilt(_sos, chunk.astype(np.float64)).astype(np.float32)

            feed = {
                "waveform": x_lim[None, None, :],
                "cutoff": np.array([[c_norm]], dtype=np.float32),
            }
            pred = self._session.run(None, feed)[0][0, 0, :].astype(np.float64)  # [T]

            chunk_out = blend_strength * pred + (1.0 - blend_strength) * chunk.astype(np.float64)

            # Overlap-Add mit linearer Crossfade (§G136 deterministisch):
            # Chunk = exakt 2 Hops → Fade-in über ersten Hop, Fade-out über
            # zweiten Hop; im Überlapp summieren sich die Rampen zu 1.
            valid = min(len(chunk_out), n - pos)
            is_first = pos == 0
            is_last = pos + self._V5_HOP >= n
            w = np.ones(valid, dtype=np.float64)
            if not is_first:
                _n_in = min(self._V5_HOP, valid)
                w[:_n_in] = ramp[:_n_in]
            if not is_last:
                _n_out = min(self._V5_HOP, valid)
                w[valid - _n_out :] = 1.0 - ramp[-_n_out:]
            out_accum[pos : pos + valid] += chunk_out[:valid] * w
            out_weight[pos : pos + valid] += w

            pos += self._V5_HOP

        out = np.where(out_weight > 1e-9, out_accum / np.maximum(out_weight, 1e-9), audio_22k)
        return cast(np.ndarray, (np.asarray(out, dtype=np.float64)))

    # ── Interne Hilfsfunktionen ────────────────────────────────────────────

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Einfaches Resampling via linearer Interpolation (keine externe Abhängigkeit)."""
        if orig_sr == target_sr:
            return audio.copy()

        ratio = target_sr / orig_sr
        n_out = int(len(audio) * ratio)
        indices = np.arange(n_out) / ratio
        lo = np.floor(indices).astype(int)
        hi = np.clip(lo + 1, 0, len(audio) - 1)
        lo = np.clip(lo, 0, len(audio) - 1)
        frac = indices - lo
        return (1.0 - frac) * audio[lo] + frac * audio[hi]  # type: ignore[no-any-return]

    def _audio_to_mel(self, audio: np.ndarray) -> np.ndarray:
        """Konvertiert Audio zu logarithmiertem Mel-Spektrogramm via STFT."""
        # Hann-Fenster
        window = np.hanning(self._N_FFT)

        # Mel-Filterbank (einfache Annäherung: Dreiecksfilter auf FFT-Bins)
        n_fft_out = self._N_FFT // 2 + 1
        mel_basis = self._build_mel_basis(self._SR, self._N_FFT, self._N_MELS)

        # STFT
        n_frames = 1 + (len(audio) - self._N_FFT) // self._HOP_LENGTH
        if n_frames < 1:
            audio = np.pad(audio, (0, self._N_FFT))
            n_frames = 1

        spec = np.zeros((n_fft_out, n_frames), dtype=np.float32)
        for i in range(n_frames):
            start = i * self._HOP_LENGTH
            frame = audio[start : start + self._N_FFT] * window
            spec[:, i] = np.abs(np.fft.rfft(frame, n=self._N_FFT))

        # Mel
        mel = mel_basis @ spec

        # Log
        mel_db = 20 * np.log10(np.maximum(mel, 1e-8))
        # Normalisierung auf [0, 1]
        mel_norm = np.clip((mel_db + 80.0) / 80.0, 0.0, 1.0)

        return mel_norm.astype(np.float32)  # type: ignore[no-any-return]

    def _mel_to_audio(self, mel_norm: np.ndarray) -> np.ndarray:
        """Griffin-Lim: Rekonstruiert Audio aus Mel-Spektrogramm via iterative Phasenschätzung."""
        # Denormalisieren
        mel_db = mel_norm * 80.0 - 80.0
        mel = np.power(10.0, mel_db / 20.0)

        # Inverse Mel — einfache Annäherung via Pseudo-Inverse
        mel_basis = self._build_mel_basis(self._SR, self._N_FFT, self._N_MELS)
        mel_basis_pinv = np.linalg.pinv(mel_basis)

        mag_spec = mel_basis_pinv @ mel
        mag_spec = np.maximum(mag_spec, 0.0)

        # Griffin-Lim (10 Iterationen)
        n_frames = mag_spec.shape[1]
        angles = np.random.uniform(0, 2 * np.pi, mag_spec.shape)
        window = np.hanning(self._N_FFT)

        for _ in range(10):
            # ISTFT mit aktuellen Phasen
            stft = mag_spec * np.exp(1j * angles)
            audio = np.zeros((n_frames - 1) * self._HOP_LENGTH + self._N_FFT, dtype=np.float64)
            weight = np.zeros_like(audio)

            for i in range(n_frames):
                frame = np.fft.irfft(stft[:, i], n=self._N_FFT)
                start = i * self._HOP_LENGTH
                audio[start : start + self._N_FFT] += frame * window
                weight[start : start + self._N_FFT] += window**2

            audio /= np.maximum(weight, 1e-8)

            # STFT für neue Phasen
            new_stft = np.zeros_like(stft, dtype=complex)
            for i in range(n_frames):
                start = i * self._HOP_LENGTH
                frame = audio[start : start + self._N_FFT] * window
                new_stft[:, i] = np.fft.rfft(frame, n=self._N_FFT)
            angles = np.angle(new_stft)

        # Letzte ISTFT
        audio = np.zeros((n_frames - 1) * self._HOP_LENGTH + self._N_FFT, dtype=np.float64)
        weight = np.zeros_like(audio)
        denorm = np.max(mag_spec) + 1e-8

        for i in range(n_frames):
            frame = np.fft.irfft(mag_spec[:, i] / denorm * np.exp(1j * angles[:, i]), n=self._N_FFT)
            start = i * self._HOP_LENGTH
            audio[start : start + self._N_FFT] += frame * window
            weight[start : start + self._N_FFT] += window**2

        audio /= np.maximum(weight, 1e-8)
        audio = np.clip(audio, -1.0, 1.0)

        return cast(np.ndarray, audio.astype(np.float32))

    def _process_segments(self, mel_input: np.ndarray) -> np.ndarray:
        """ONNX-Inferenz in überlappenden 256×256-Segmenten."""
        n_mels, n_frames_total = mel_input.shape
        target_frames = self._N_FRAMES

        if n_frames_total <= target_frames:
            # Einzelnes Segment, padden
            padded = np.zeros((self._N_MELS, target_frames), dtype=np.float32)
            padded[:, :n_frames_total] = mel_input[:, :target_frames]
            return self._infer_single(padded)[:, :n_frames_total]

        # Überlappende Segmente
        hop = target_frames // 2
        n_segments = max(1, 1 + (n_frames_total - target_frames) // hop)

        output = np.zeros((self._N_MELS, n_frames_total), dtype=np.float32)
        weight = np.zeros((self._N_MELS, n_frames_total), dtype=np.float32)
        hann_window = np.hanning(target_frames).astype(np.float32)

        for seg_idx in range(n_segments):
            start = seg_idx * hop
            end = min(start + target_frames, n_frames_total)

            segment = np.zeros((self._N_MELS, target_frames), dtype=np.float32)
            seg_len = end - start
            segment[:, :seg_len] = mel_input[:, start:end]

            recon_seg = self._infer_single(segment)

            out_len = min(target_frames, n_frames_total - start)
            window_2d = hann_window[:out_len][np.newaxis, :]
            output[:, start : start + out_len] += recon_seg[:, :out_len] * window_2d
            weight[:, start : start + out_len] += window_2d

        output /= np.maximum(weight, 1e-8)
        return cast(np.ndarray, output)

    def _infer_single(self, mel_segment: np.ndarray) -> np.ndarray:
        """Einzelne ONNX-Inferenz: (256, 256) → (256, 256)."""
        if self._session is None:
            return mel_segment.copy()

        # Input: (1, 1, 256, 256)
        x = mel_segment[np.newaxis, np.newaxis, :, :].astype(np.float32)

        # PLM-Active-Guard
        self._set_plm_active(True)
        try:
            result = self._session.run(["output"], {"input": x})[0]
        finally:
            self._set_plm_active(False)

        return result[0, 0, :, :]  # type: ignore[no-any-return]

    def _set_plm_active(self, active: bool) -> None:
        """Teilt dem Plugin Lifecycle Manager den aktiven Inferenz-Status mit."""
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager

            plm = get_plugin_lifecycle_manager()
            plm.set_active(self._BUDGET_NAME, active)
        except Exception as _e:
            logger.debug("%s: unkritisch exception: %s", __name__, _e)

    def _estimate_cutoff(self, audio: np.ndarray, sr: int, threshold_db: float = -60.0) -> float:
        """Schätzt die Bandbreitengrenze via FFT-Energie-Abfall."""
        n = min(4096, len(audio))
        if n < 256:
            return sr / 2.0

        spec = np.abs(np.fft.rfft(audio[:n] * np.hanning(n), n=n))
        spec_db = 20 * np.log10(spec + 1e-10)
        max_db = spec_db.max()
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)

        above = spec_db > (max_db + threshold_db)
        if not above.any():
            return sr / 2.0

        last_bin = np.where(above)[0][-1]
        return float(freqs[min(last_bin + 1, len(freqs) - 1)])

    @staticmethod
    def _build_mel_basis(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Baut eine Mel-Filterbank (Dreiecksfilter)."""
        n_fft_out = n_fft // 2 + 1
        f_min = 0.0
        f_max = sr / 2.0

        mel_min = 2595.0 * np.log10(1.0 + f_min / 700.0)
        mel_max = 2595.0 * np.log10(1.0 + f_max / 700.0)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)

        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        bin_points = np.clip(bin_points, 0, n_fft_out - 1)

        basis = np.zeros((n_mels, n_fft_out), dtype=np.float32)
        for m in range(n_mels):
            left = bin_points[m]
            center = bin_points[m + 1]
            right = bin_points[m + 2]

            for k in range(left, center):
                if center > left:
                    basis[m, k] = (k - left) / (center - left)
            for k in range(center, right):
                if right > center:
                    basis[m, k] = 1.0 - (k - center) / (right - center)

        return cast(np.ndarray, basis)


# ── Convenience-Funktion ───────────────────────────────────────────────────


def reconstruct_bandwidth(
    audio: np.ndarray,
    sr: int,
    cutoff_hz: float | None = None,
    model_path: str | None = None,
    blend_strength: float = 1.0,
) -> np.ndarray:
    """Einzeiler für die Bandbreiten-Rekonstruktion.

    Args:
        audio: Audio-Array (Mono/Stereo, float32/64).
        sr: Samplerate in Hz.
        cutoff_hz: Bandbreitengrenze (None = automatisch schätzen).
        model_path: Pfad zum ONNX-Modell (None = Default).
        blend_strength: Blend-Faktor (0=Original, 1=voll).

    Returns:
        Rekonstruiertes Audio.
    """
    plugin = BWReconstructorPlugin(model_path=model_path)
    return plugin.reconstruct(audio, sr, cutoff_hz=cutoff_hz, blend_strength=blend_strength)


# ── Pipeline-Integration ───────────────────────────────────────────────────


class BWReconstructorStage:
    """Aurik-Pipeline-Stage für Bandbreiten-Rekonstruktion.

    Kann als normale Stage in den Restorer eingehängt werden:

        from plugins.bw_reconstructor_plugin import BWReconstructorStage
        pipeline.add_stage(BWReconstructorStage(cutoff_hz=4000))
    """

    def __init__(
        self,
        cutoff_hz: float | None = None,
        model_path: str | None = None,
        blend_strength: float = 1.0,
        enabled: bool = True,
    ):
        self._plugin = BWReconstructorPlugin(model_path=model_path) if enabled else None
        self.cutoff_hz = cutoff_hz
        self.blend_strength = blend_strength
        self.enabled = enabled

    @property
    def name(self) -> str:
        return "BWReconstructor"

    @property
    def available(self) -> bool:
        return self.enabled and self._plugin is not None and self._plugin.available

    def process(self, audio: np.ndarray, sr: int, **kwargs: Any) -> dict[str, Any]:
        """Pipeline-kompatible process-Methode.

        Returns:
            {"audio": reconstructed_audio, "metadata": {...}}
        """
        if not self.available:
            return {"audio": audio, "metadata": {"bw_reconstructor": "unavailable"}}

        cutoff = kwargs.get("cutoff_hz", self.cutoff_hz)
        blend = kwargs.get("blend_strength", self.blend_strength)

        reconstructed = self._plugin.reconstruct(audio, sr, cutoff_hz=cutoff, blend_strength=blend)  # type: ignore[union-attr]

        estimated_cutoff = self._plugin._estimate_cutoff(audio.squeeze(), sr)  # type: ignore[union-attr]

        return {
            "audio": reconstructed,
            "metadata": {
                "bw_reconstructor": "applied",
                "estimated_cutoff_hz": round(estimated_cutoff, 1),
                "applied_cutoff_hz": cutoff,
                "blend_strength": blend,
            },
        }
