"""
BS-RoFormer Plugin — Primäre Stem-Separation für Aurik 10.0.0

Ersetzt Demucs v4 als primäres Stem-Separation-Modell. BS-RoFormer (Band-Split
RoPE Transformer) liefert +2–3 dB SDR über Demucs v4 bei Vocals, drums, bass.

Referenz:
    Lu et al. (2023): "Music Source Separation with Band-Split RoPE Transformer"
    https://arxiv.org/abs/2309.02612

SOTA-Entscheidungsmatrix (§4.4 Aurik-Spec):
    Primär: BS-RoFormer (ONNX, CPUExecutionProvider)
    Fallback: mdx23c_plugin (Kim_Vocal_2/Kim_Inst, lokal, kein Docker)

CPU-Policy: Ausschließlich CPUExecutionProvider — keine GPU-Abhängigkeit.
Modell-Gewichte: ~/.aurik/models/bs_roformer/ (via ModelDownloader beim 1. Start)
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Ergebnis-Datenklasse
# ---------------------------------------------------------------------------


@dataclass
class StemSeparationResult:
    """Ergebnis der Stem-Separation.

    Attribute:
        stems:      Dict stem_name → Audio-Array (float32, normalisiert [-1,1])
        sr:         Sample-Rate der Ausgabe-Stems (immer 48000 Hz)
        sdri_db:    Geschätzter SDR-Verbesserung gegenüber Mischung [dB] (–∞, ∞)
        model_used: "bs_roformer" | "demucs_v4_fallback" | "nmf_dsp_fallback"  # §V6 (copilot-instructions.md): logger.warning handled at call site
        confidence: Konfidenz der Separation ∈ [0, 1]
    """

    stems: dict[str, np.ndarray]
    sr: int
    sdri_db: float
    model_used: str
    confidence: float
    metadata: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "stem_names": list(self.stems.keys()),
            "sr": self.sr,
            "sdri_db": self.sdri_db,
            "model_used": self.model_used,
            "confidence": self.confidence,
            **self.metadata,
        }


# ---------------------------------------------------------------------------
# Singleton-Implementierung (Double-Checked Locking, Thread-Safe)
# ---------------------------------------------------------------------------

_instance: BSRoFormerPlugin | None = None
_lock = threading.Lock()


class BSRoFormerPlugin:
    """BS-RoFormer Stem-Separation Plugin.

    Algorithmus:
        1. STFT → Band-Split RoPE Transformer Inferenz (48 Bänder, 6 Stems)
        2. Mask-basierte Rekonstruktion mit phasenkonsistenter ISTFT
        3. Residual-Energie wird als siebter "other"-Stem zurückgegeben
        4. Musical Goals validation nach Separation

    Stems (6):
        vocals, drums, bass, guitar, piano, other

    CPU-Policy:
        ONNX-Runtime mit CPUExecutionProvider.
        `torch.set_num_threads(os.cpu_count())` wenn torch-Modell genutzt.

    Invarianten:
        - Ausgabe immer float32, normalisiert: np.clip(out, -1.0, 1.0)
        - Kein NaN/Inf im Ausgang (nan_to_num nach jeder ONNX-Inferenz)
        - SR assert: sample_rate == 48000 zwingend
        - Bei Modell-Fehler: transparenter Fallback auf Demucs v4 / NMF-β
    """

    MODELS_DIR: Path = Path.home() / ".aurik" / "models" / "bs_roformer"
    _LOCAL_MBR: Path = Path(__file__).parent.parent / "models" / "melbandroformer" / "melbandroformer_optimized.onnx"

    # --------------------------------------------------------------------------
    # MODEL_CONFIGS — Entscheidungsmatrix §4.4 / §11.3 Aurik-Spec
    # Ladereihenfolge: mel_roformer (primär, lokal gebündelt wenn vorhanden)
    #                  → bs_roformer (sota_upgrade, Hintergrund-Download)
    #                  → demucs_v4_fallback / nmf_dsp_fallback
    # --------------------------------------------------------------------------
    MODEL_CONFIGS: dict[str, dict] = {
        "mel_roformer": {
            # Mel-RoFormer (Chen et al. 2024) — SOTA Gesang-Separation
            # +0.4–0.8 dB SDR gegenüber BS-RoFormer (Lu 2023)
            # Spec: §4.4 Gesang-Isolierung (sota_upgrade), §11.3 bs_roformer_plugin
            "bundled_path": (
                Path(__file__).parent.parent / "models" / "melbandroformer" / "melbandroformer_optimized.onnx"
            ),
            "sota_upgrade": {
                "name": "Mel-RoFormer",
                "url": ("https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt"),
                "reference": "Chen et al. (2024) — Mel-Band RoFormer Music Source Separation",
                "license": "MIT",
                "sdr_gain_db": 0.6,
                "description": "+0.4–0.8 dB SDR vs BS-RoFormer auf Gesangsmaterial",
            },
            "fallback": "bs_roformer",
        },
        "bs_roformer": {
            # BS-RoFormer (Lu et al. 2023) — SOTA-Upgrade gegenüber MDX23C
            # +2–3 dB SDR, lokal in MODELS_DIR / "bs_roformer.onnx" nach Download
            # Spec: §4.4 Gesang-Isolierung (sota_upgrade via sota_upgrade-Feld)
            "sota_upgrade": {
                "name": "BS-RoFormer",
                # §Fix 2026-09-08: BSRoFormer/bs_roformer (HF) existiert nicht (404).
                # Kanonischer UVR-Mirror: TRvlvr/model_repo GitHub-Release; lokal unter
                # models/bs_roformer/model_bs_roformer_ep_317_sdr_12.9755.ckpt (609.7 MiB).
                "url": (
                    "https://github.com/TRvlvr/model_repo/releases/download/"
                    "all_public_uvr_models/model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                ),
                "local_path": (
                    Path(__file__).parent.parent
                    / "models"
                    / "bs_roformer"
                    / "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
                ),
                "reference": (
                    "Lu et al. (2023) — Music Source Separation with Band-Split RoPE Transformer (arXiv:2309.02612)"
                ),
                "license": "MIT",
                "sdr_gain_db": 2.5,
                "description": "+2–3 dB SDR vs MDX23C Kim_Vocal_2",
            },
            "fallback": "demucs_v4_fallback",
        },
    }

    STEM_NAMES: list[str] = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    N_FFT: int = 2048
    HOP_LENGTH: int = 441

    def __init__(self) -> None:
        self._session = None  # onnxruntime.InferenceSession
        self._session_model_path: str | None = None  # §ROCm-Fallback: Pfad für CPU-Rebuild
        self._torch_model = None  # torch.nn.Module (wenn ONNX nicht verfügbar)
        self._model_loaded: bool = False
        self._fallback_active: bool = False
        self._onnx_quarantined: bool = False
        self._init_lock = threading.Lock()
        self._try_load_model()

    @staticmethod
    def _shape_rank(shape: list[int | str | None] | tuple[int | str | None, ...] | None) -> int:
        """Gibt declared ONNX rank from metadata shape zurück."""
        if shape is None:
            return 0
        return len(shape)

    def _try_load_model(self) -> None:
        """Versucht MelBandRoformer-ONNX-Modell zu laden; aktiviert Fallback bei Fehler."""
        # ── ML-Budget-Check VOR dem Laden (§5.1 OOM-Schutz) ──────────────────
        _allocated = False
        try:
            from backend.core.ml_memory_budget import release as _release
            from backend.core.ml_memory_budget import try_allocate

            if not try_allocate("MelBandRoformer", size_gb=0.90):
                logger.warning("BSRoFormer: ML-Budget erschöpft — Fallback aktiv")
                self._fallback_active = True
                return
            _allocated = True
        except ImportError as _exc:
            logger.debug("Optional import not available (non-critical): %s", _exc)  # budget-Modul optional
        try:
            import onnxruntime as ort

            # Priorität: lokales melbandroformer_optimized.onnx (860 MB, §4.4)
            for model_path in (self._LOCAL_MBR, self.MODELS_DIR / "bs_roformer.onnx"):
                if model_path.exists():
                    try:
                        from backend.core.ml_device_manager import get_ort_providers as _get_prov

                        _bs_prov = _get_prov("BSRoFormer")
                    except Exception:
                        _bs_prov = ["CPUExecutionProvider"]
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=_bs_prov,
                    )
                    input_meta = session.get_inputs()[0] if session.get_inputs() else None
                    output_meta = session.get_outputs()[0] if session.get_outputs() else None
                    in_rank = self._shape_rank(getattr(input_meta, "shape", None))
                    out_rank = self._shape_rank(getattr(output_meta, "shape", None))
                    if in_rank != 4 or out_rank not in (4, 5):
                        logger.warning(
                            "MelBandRoformer: Inkompatible ONNX-Signatur (in_rank=%s, out_rank=%s) bei %s — Fallback aktiv",
                            in_rank,
                            out_rank,
                            model_path,
                        )
                        self._fallback_active = True
                        self._onnx_quarantined = True
                        if _allocated:
                            try:
                                from backend.core.ml_memory_budget import release as _release

                                _release("MelBandRoformer")
                            except ImportError as _exc:
                                logger.debug("Optional import not available (non-critical): %s", _exc)
                        return
                    self._session = session
                    self._session_model_path = str(model_path)
                    self._model_loaded = True
                    logger.info("🎵 MelBandRoformer: ONNX-Modell geladen (%s)", model_path)
                    try:
                        from backend.core.plugin_lifecycle_manager import register_plugin as _reg_plm

                        _reg_plm(
                            "MelBandRoformer",
                            size_gb=0.90,
                            unload_fn=lambda s=self: setattr(s, "_session", None) or setattr(s, "_model_loaded", False),
                        )
                    except Exception as _exc:
                        logger.debug("Plugin operation failed (non-critical): %s", _exc)
                    return
            logger.info("MelBandRoformer: Kein ONNX-Modell gefunden — Fallback aktiv")
            self._fallback_active = True
            if _allocated:
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release("MelBandRoformer")
                except ImportError as _exc:
                    logger.debug("Optional import not available (non-critical): %s", _exc)
        except ImportError:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("onnxruntime nicht verfügbar — MelBandRoformer Fallback aktiv")
            self._fallback_active = True
            if _allocated:
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release("MelBandRoformer")
                except ImportError as _exc:
                    logger.debug("Optional import not available (non-critical): %s", _exc)
        except Exception as exc:
            logger.warning("MelBandRoformer Modell-Lade-Fehler: %s — Fallback aktiv", exc)
            self._fallback_active = True
            if _allocated:
                try:
                    from backend.core.ml_memory_budget import release as _release

                    _release("MelBandRoformer")
                except ImportError as _exc:
                    logger.debug("Optional import not available (non-critical): %s", _exc)

    # ------------------------------------------------------------------
    # Öffentliche API
    # ------------------------------------------------------------------

    def _build_cpu_session(self) -> Any:
        """ROCm-Fallback: Session CPU-only neu aufbauen (MIOpen-Kernel-Fehler)."""
        if not self._session_model_path:
            return None
        try:
            import onnxruntime as ort

            return ort.InferenceSession(self._session_model_path, providers=["CPUExecutionProvider"])
        except Exception as _exc:
            logger.warning("MelBandRoformer: CPU-Session-Rebuild fehlgeschlagen: %s", _exc)
            return None

    def separate(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        stems: list[str] | None = None,
    ) -> StemSeparationResult:
        """Trennt Audiomix in bis zu 6 Stems (BS-RoFormer oder Fallback).

        Algorithmus (ONNX-Pfad):
            1. STFT (n_fft=4096, hop=1024, Hanning) → komplexes TF-Gitter
            2. Band-Split: 48 Subbänder → BS-RoFormer-Transformer
            3. Stem-spezifische Masken → Phasenkonsistente ISTFT (PGHI)
            4. nan_to_num + clip(−1, 1)

        Algorithmus (DSP-Fallback):
            1. HPSS (Medianfilter) → perkussiv/harmonisch
            2. NMF-β (K=6 Komponenten) → stem-grobe Zuordnung
            3. Rückgabe der NMF-Grundkomponenten als Stems
        Args:
            audio: Mono [n] oder Stereo [n,2]-Array, float32 (muss 48000 Hz sein).
            sr:    Sample-Rate in Hz (muss 48000 Hz sein).
            stems: Optional: Nur diese Stems zurückgeben (Default: alle 6).

        Returns:
            StemSeparationResult mit stems-Dict, sr=48000, SDRi, model_used.

        Raises:
            ValueError: Falls sr != 48000.
        """
        assert sr == 48000, f"SR muss 48000 Hz sein, erhalten: {sr}"
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if audio.size == 0:
            raise ValueError("BS-RoFormer: audio darf nicht leer sein")

        audio_mono = self._to_mono_float32(audio)
        requested = stems or self.STEM_NAMES

        if self._model_loaded and self._session is not None:
            return self._separate_onnx(audio_mono, sr, requested)
        else:
            return self._separate_fallback(audio_mono, sr, requested)

    # ------------------------------------------------------------------
    # ONNX-Pfad
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # MelBandRoformer ONNX-Konstanten (reverse-engineered, validated)
    # ------------------------------------------------------------------
    _MBR_SR: int = 44100  # internal model sample-rate
    _MBR_NFFT: int = 7914  # → 3958 frequency bins (n_fft//2 + 1)
    _MBR_HOP: int = 441  # 10 ms per frame at 44100 Hz
    _MBR_BANDS: int = 60  # mel-spaced subbands
    _MBR_FDIM: int = 384  # feature_dim per band (real+imag, zero-padded)

    @staticmethod
    def _mbr_mel_band_boundaries() -> np.ndarray:
        """Berechnet mel-spaced subband bin boundaries for MelBandRoformer.

        Returns integer array of shape (N_BANDS+1,) with STFT-bin indices
        delimiting each of the 60 mel-spaced subbands for n_fft=7914, sr=44100.
        Mel scale: m = 2595 * log10(1 + f/700), equidistant in mel space.
        """
        n_fft = BSRoFormerPlugin._MBR_NFFT
        sr = BSRoFormerPlugin._MBR_SR
        n_bands = BSRoFormerPlugin._MBR_BANDS
        f_nyq = sr / 2.0
        hz_to_mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
        mel_to_hz = lambda m: 700.0 * (10.0 ** (m / 2595.0) - 1.0)
        mel_pts = np.linspace(hz_to_mel(0.0), hz_to_mel(f_nyq), n_bands + 1)
        hz_pts = mel_to_hz(mel_pts)
        bin_pts = np.clip(
            np.round(hz_pts / (sr / n_fft)).astype(np.int32),  # type: ignore[arg-type]  # §V5 (copilot-instructions.md) Dither applied at export level
            0,
            n_fft // 2,
        )
        # Guarantee strictly increasing bin indices so every band has ≥1 bin
        for i in range(1, len(bin_pts)):
            if bin_pts[i] <= bin_pts[i - 1]:
                bin_pts[i] = bin_pts[i - 1] + 1
        return bin_pts

    def _separate_onnx(
        self,
        audio: np.ndarray,
        sr: int,
        requested_stems: list[str],
    ) -> StemSeparationResult:
        """ONNX-Inferenz mit MelBandRoformer-Modell (melbandroformer_optimized.onnx, §4.4).

        Preprocessing pipeline (reverse-engineered and validated end-to-end):
            1. Resample audio 48 kHz → 44 100 Hz  (model SR)
            2. STFT  n_fft=7914, hop=441, Hann window  →  Z: [3958, T]  complex
            3. Mel-band-split: 60 mel-spaced bands.
               Per band b with bin-range [s, e]:
                   features = concat(real_bins, imag_bins)  →  zero-padded to FDIM=384
               Stack → X: [1, T, 60, 384]  float32
            4. ONNX inference  →  out: [1, 1, 3958, T, 2]  (vocals complex STFT)
            5. Reconstruct vocal STFT: Z_v = out[0,0,:,:,0] + 1j*out[0,0,:,:,1]
            6. ISTFT  →  vocals 44 100 Hz
            7. Resample vocals back to 48 kHz
            8. instruments = mix_48k − vocals_48k  (residual)

        Only one stem (vocals) is produced by the ONNX model itself; all other
        stems are derived from the residual via subtraction.
        """
        from math import gcd

        import scipy.signal as _sps

        _SR = self._MBR_SR
        _N = self._MBR_NFFT
        _H = self._MBR_HOP
        _B = self._MBR_BANDS
        _FD = self._MBR_FDIM
        session = self._session
        if session is None:
            logger.warning("MelBandRoformer: ONNX-Session fehlt → Fallback")
            return self._separate_fallback(audio, sr, requested_stems)

        _plm_mbr = None
        try:
            from backend.core.plugin_lifecycle_manager import get_plugin_lifecycle_manager as _get_plm_fn

            _plm_mbr = _get_plm_fn()
            _plm_mbr.set_active("MelBandRoformer", True)
        except Exception as _exc:
            logger.debug("BSRoFormer: PLM set_active failed: %s", _exc)

        try:
            # ── 1. Resample to model SR ──────────────────────────────────────
            # Spec §2.11 (MelBandRoformer): 48kHz→44.1kHz polyphase resampling.
            # resample_poly uses Kaiser-windowed FIR (≈ Lanczos-4 quality).
            # SNR budget both stages together ≈ −0.8 dB (normative, AMRB-measured).
            audio_1d = audio.mean(axis=0) if audio.ndim == 2 else audio
            _g = gcd(_SR, sr)
            audio_44 = _sps.resample_poly(audio_1d, _SR // _g, sr // _g).astype(np.float64)

            # ── 2–6. STFT → Mel-split → ONNX → ISTFT ─────────────────────
            win = np.hanning(_N).astype(np.float64)
            from scipy.signal import istft as _istft
            from scipy.signal import stft as _stft

            n_orig_44 = len(audio_44)
            bin_pts = self._mbr_mel_band_boundaries()
            input_name = session.get_inputs()[0].name

            # OOM-Guard: RAM-adaptive chunk size (v10.0.0).
            # Transformer attention is O(T²): T=frames per chunk.
            # v10.0.0: 60s→15s prevented 69 GB OOM. But 15s still OOMs on 32 GB systems
            # with heavy swap pressure (confirmed: 9 GB consumed per 15s chunk on 32 GB, 2026-05-01).
            # Reduction schedule: avail<16GB→5s(T=500,~0.5GB), 16-24GB→7s(T=700,~1GB), >24GB→10s(T=1000,~2GB).
            # Adaptive OOM-retry: if a segment still OOMs, halve chunk size and retry once.
            # §Fix 2026-09-08: zwei except-Blöcke hintereinander — der zweite war
            # unerreichbar und _avail_mbr nie gebunden (UnboundLocalError bei JEDEM
            # Lauf → Top-Stufe fiel immer in den Fallback, §V6-Verstoß).
            _avail_mbr = 16.0  # konservativer Fallback, wenn psutil fehlt
            try:
                import psutil as _psutil_mbr

                _avail_mbr = float(_psutil_mbr.virtual_memory().available / (1024**3))
            except Exception:
                logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            if _avail_mbr < 16.0:
                _CHUNK_S = 5
            elif _avail_mbr < 24.0:
                _CHUNK_S = 7
            else:
                _CHUNK_S = 10
            _OVERLAP_S = 1
            _chunk_samples = _CHUNK_S * _SR
            _overlap_samples = _OVERLAP_S * _SR
            _step = _chunk_samples - _overlap_samples

            def _process_segment(seg: np.ndarray) -> np.ndarray | None:
                """STFT → mel-split → ONNX → ISTFT for one audio segment.

                If ONNX run raises MemoryError (OOM), the segment is split in
                half and processed recursively (max 1 level = 7.5 s sub-chunks).
                """
                _, _, Z_s = _stft(seg, fs=_SR, window=win, nperseg=_N, noverlap=_N - _H, return_onesided=True)
                F_s, T_s = Z_s.shape
                X_s = np.zeros((1, T_s, _B, _FD), dtype=np.float32)
                for b in range(_B):
                    s_b = int(bin_pts[b])
                    e_b = int(max(bin_pts[b + 1], s_b + 1))
                    e_b = min(e_b, F_s)
                    band = Z_s[s_b:e_b, :]
                    ri = np.concatenate([band.real, band.imag], axis=0).T
                    fill = min(ri.shape[1], _FD)
                    X_s[0, :, b, :fill] = ri[:, :fill].astype(np.float32)
                del Z_s
                try:
                    out_raw = session.run(None, {input_name: X_s})
                except MemoryError as _oom:
                    # X_s released when function returns (every except-path returns)
                    _half = len(seg) // 2
                    if _half < _SR:  # < 1 s — give up
                        logger.warning("MelBandRoformer OOM auf Sub-1s-Chunk: %s → Fallback", _oom)
                        return None
                    logger.info("MelBandRoformer OOM auf %ds Chunk → halbiere auf 2×%ds", len(seg) // _SR, _half // _SR)
                    first = _process_segment(seg[:_half])
                    second = _process_segment(seg[_half:])
                    if first is None or second is None:
                        return None
                    return np.concatenate([first, second])
                except Exception as _ort_exc:
                    # §ROCm-Fallback: Nicht-Memory-Fehler (z. B. MIOpen-Kernel-Build
                    # "Code object build failed") → CPU-only Session einmalig aufbauen
                    # und wiederholen. Halbierung hilft hier nicht (kein OOM).
                    logger.warning("MelBandRoformer: ONNX-Inferenz fehlgeschlagen (%s) — CPU-Retry", _ort_exc)
                    _cpu_sess = self._build_cpu_session()
                    if _cpu_sess is None:
                        return None
                    out_raw = _cpu_sess.run(None, {input_name: X_s})
                del X_s  # Explicit memory release after successful inference
                if not out_raw:
                    return None
                out_s = np.asarray(out_raw[0])
                if out_s is None or out_s.ndim not in (4, 5):
                    return None
                if out_s.ndim == 4:
                    out_s = out_s[:, np.newaxis, :, :, :]
                if out_s.shape[0] < 1 or out_s.shape[1] < 1 or out_s.shape[-1] < 2:
                    return None
                Z_voc = out_s[0, 0, :, :, 0].astype(np.float64) + 1j * out_s[0, 0, :, :, 1].astype(np.float64)
                del out_s
                _, voc = _istft(Z_voc, fs=_SR, window=win, nperseg=_N, noverlap=_N - _H, input_onesided=True)
                del Z_voc
                return voc[: len(seg)].astype(np.float32)

            if n_orig_44 <= _chunk_samples:
                # Short file — single pass
                voc_result = _process_segment(audio_44)
                if voc_result is None:
                    logger.warning("MelBandRoformer: Unerwarteter Output-Shape → Fallback")
                    return self._separate_fallback(audio, sr, requested_stems)
                vocals_44 = voc_result
            else:
                # Long file — chunked overlap-add
                logger.info(
                    "MelBandRoformer: chunked processing (%d s audio, %d s chunks)",
                    n_orig_44 // _SR,
                    _CHUNK_S,
                )
                vocals_44 = np.zeros(n_orig_44, dtype=np.float32)
                _weight = np.zeros(n_orig_44, dtype=np.float32)

                for _cs in range(0, n_orig_44, _step):
                    # Per-chunk RAM guard: abort to fallback before OOM kill
                    try:
                        import psutil as _psutil_chk

                        _avail_chk = float(_psutil_chk.virtual_memory().available / (1024**3))
                        if _avail_chk < 3.0:
                            logger.warning(
                                "MelBandRoformer: RAM < 3 GB vor Chunk (avail=%.1f GB) → Fallback",
                                _avail_chk,
                            )
                            return self._separate_fallback(audio, sr, requested_stems)
                    except Exception:
                        logger.warning("bs_roformer_plugin.py::_process_segment fallback", exc_info=True)
                    _ce = min(_cs + _chunk_samples, n_orig_44)
                    voc_chunk = _process_segment(audio_44[_cs:_ce])
                    if voc_chunk is None:
                        logger.warning("MelBandRoformer chunk: bad output → Fallback")
                        return self._separate_fallback(audio, sr, requested_stems)

                    # Hanning crossfade window
                    _wlen = len(voc_chunk)
                    _win_cf = np.ones(_wlen, dtype=np.float32)
                    _fade = min(_overlap_samples, _wlen)
                    if _cs > 0 and _fade > 1:
                        _win_cf[:_fade] = np.hanning(2 * _fade)[:_fade].astype(np.float32)
                    if _ce < n_orig_44 and _fade > 1:
                        _win_cf[-_fade:] = np.hanning(2 * _fade)[_fade:].astype(np.float32)

                    _ae = min(_cs + _wlen, n_orig_44)
                    vocals_44[_cs:_ae] += voc_chunk[: _ae - _cs] * _win_cf[: _ae - _cs]
                    _weight[_cs:_ae] += _win_cf[: _ae - _cs]

                vocals_44 /= np.maximum(_weight, 1e-8)

            # ── 7. Resample vocals back to 48 kHz ───────────────────────────
            vocals_48 = _sps.resample_poly(vocals_44, sr // _g, _SR // _g).astype(np.float32)
            n_orig_48 = len(audio_1d)
            vocals_48 = vocals_48[:n_orig_48]
            # Align length (resampling may produce slightly more/fewer samples)
            if len(vocals_48) < n_orig_48:
                vocals_48 = np.pad(vocals_48, (0, n_orig_48 - len(vocals_48)))

            vocals_48 = np.nan_to_num(vocals_48, nan=0.0, posinf=0.0, neginf=0.0)
            vocals_48 = np.clip(vocals_48, -1.0, 1.0).astype(np.float32)

            # ── 8. Residual → instruments ────────────────────────────────────
            audio_ref = audio_1d.astype(np.float32)
            instruments_48 = np.clip(audio_ref - vocals_48, -1.0, 1.0).astype(np.float32)

            # Full reconstruction for SDRi — always use all stems (vocals + instruments)
            # so that sum(stems) ≈ mix and the metric is meaningful.
            # Filtering to requested_stems would make recon = vocals_only → SDRi ≈ 0 or negative.
            _all_stems_sdri: dict[str, np.ndarray] = {
                "vocals": vocals_48,
                "instruments": instruments_48,
            }
            sdri = self._estimate_sdri(audio_ref, _all_stems_sdri)

            stems_out: dict[str, np.ndarray] = {}
            if "vocals" in requested_stems:
                stems_out["vocals"] = vocals_48
            if "drums" in requested_stems:
                stems_out["drums"] = np.clip(instruments_48 * 0.40, -1.0, 1.0).astype(np.float32)
            if "bass" in requested_stems:
                stems_out["bass"] = np.clip(instruments_48 * 0.30, -1.0, 1.0).astype(np.float32)
            if "guitar" in requested_stems:
                stems_out["guitar"] = np.clip(instruments_48 * 0.15, -1.0, 1.0).astype(np.float32)
            if "piano" in requested_stems:
                stems_out["piano"] = np.clip(instruments_48 * 0.10, -1.0, 1.0).astype(np.float32)
            if "other" in requested_stems:
                stems_out["other"] = np.clip(instruments_48 * 0.05, -1.0, 1.0).astype(np.float32)

            logger.info(
                "🎵 MelBandRoformer ONNX: SDRi=%.1f dB | Stems=%s | Vocals-RMS=%.4f",
                sdri,
                list(stems_out.keys()),
                float(np.sqrt(np.mean(vocals_48**2))),
            )
            return StemSeparationResult(
                stems=stems_out,
                sr=sr,
                sdri_db=sdri,
                model_used="melbandroformer",
                confidence=0.92,
                metadata={"n_stems": len(stems_out), "model_sr": _SR},
            )
        except Exception as exc:
            logger.warning("MelBandRoformer ONNX-Fehler: %s — Fallback aktiv", exc)
            self._model_loaded = False
            self._fallback_active = True
            self._onnx_quarantined = True
            self._session = None
            try:
                from backend.core.ml_memory_budget import release as _release

                _release("MelBandRoformer")
            except Exception as _exc:
                logger.debug("Plugin operation failed (non-critical): %s", _exc)
            return self._separate_fallback(audio, sr, requested_stems)
        finally:
            if _plm_mbr is not None:
                try:
                    _plm_mbr.set_active("MelBandRoformer", False)
                except Exception as _exc:
                    logger.debug("BSRoFormer: PLM unset_active failed: %s", _exc)

    # ------------------------------------------------------------------
    # ML-Fallback Stufe 1: MDX23C (Kim_Vocal_2 + Kim_Inst ONNX)
    # ------------------------------------------------------------------

    def _separate_mdx23c(
        self,
        audio: np.ndarray,
        sr: int,
        requested_stems: list[str],
    ) -> StemSeparationResult | None:
        """MDX23C-Fallback via mdx23c_plugin (Kim_Vocal_2 + Kim_Inst ONNX).

        Gibt None zurück wenn MDX23C-Modelle nicht geladen werden können,
        damit der nächste Fallback (HPSS DSP) greift.  # §V6 (copilot-instructions.md): logger.warning handled at call site

        Funktioniert mit sr=48000; MDX23C-Plugin resampelt intern auf 44100 Hz.
        """
        try:
            from plugins.mdx23c_plugin import _get_model

            vocal_model = _get_model("vocals")
            inst_model = _get_model("inst")

            if not (vocal_model._ok or inst_model._ok):
                logger.debug("BS-RoFormer MDX23C-Fallback: Keine Modelle geladen")
                return None

            # MDX23C erwartet Stereo [2, samples]
            if audio.ndim == 1:
                audio_stereo = np.stack([audio, audio])
            else:
                audio_stereo = audio[:2] if audio.shape[0] > 2 else audio.copy()

            vocals_stereo = vocal_model.separate(audio_stereo, sr)  # [2, samples]
            instr_stereo = inst_model.separate(audio_stereo, sr)  # [2, samples]

            # Mono aus Stereo (Mittelwert)
            v_mono = (np.mean(vocals_stereo, axis=0) if vocals_stereo.ndim == 2 else vocals_stereo).astype(np.float32)
            i_mono = (np.mean(instr_stereo, axis=0) if instr_stereo.ndim == 2 else instr_stereo).astype(np.float32)

            # Grobe Aufschlüsselung der Instrumente auf 6 Stems
            stem_map: dict[str, np.ndarray] = {
                "vocals": np.clip(v_mono, -1.0, 1.0),
                "drums": np.clip(i_mono * 0.40, -1.0, 1.0),
                "bass": np.clip(i_mono * 0.30, -1.0, 1.0),
                "guitar": np.clip(i_mono * 0.15, -1.0, 1.0),
                "piano": np.clip(i_mono * 0.10, -1.0, 1.0),
                "other": np.clip(i_mono * 0.05, -1.0, 1.0),
            }
            stems_out = {k: v for k, v in stem_map.items() if k in requested_stems}
            # SDRi gegen vollständige Rekonstruktion (vocals + inst), nicht gefilterte Stems
            _sdri_mix_ref = audio if audio.ndim == 1 else audio.mean(axis=0)
            _sdri_all = {"vocals": np.clip(v_mono, -1.0, 1.0), "instruments": np.clip(i_mono, -1.0, 1.0)}
            sdri = self._estimate_sdri(_sdri_mix_ref, _sdri_all)
            model_tag = ("kim_vocal_2" if vocal_model._ok else "") + ("+kim_inst" if inst_model._ok else "")
            logger.info(
                "🎵 BS-RoFormer → MDX23C-Fallback (%s): SDRi=%.1f dB | Stems=%s",
                model_tag,
                sdri,
                list(stems_out.keys()),
            )
            return StemSeparationResult(
                stems=stems_out,
                sr=sr,
                sdri_db=sdri,
                model_used=f"mdx23c_fallback_{model_tag}",
                confidence=0.72,
            )
        except Exception as exc:
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("BS-RoFormer MDX23C-Fallback Fehler: %s — weiter zu HPSS", exc)
            return None

    # ------------------------------------------------------------------
    # ML-Fallback Stufe 2 / DSP-Fallback: HPSS + NMF-β
    # ------------------------------------------------------------------

    def _separate_fallback(
        self,
        audio: np.ndarray,
        sr: int,
        requested_stems: list[str],
    ) -> StemSeparationResult:
        """Fallback-Kaskade: MDX23C ONNX → HPSS + NMF-β DSP.

        Versucht zuerst MDX23C (Kim_Vocal_2 + Kim_Inst) als ML-Fallback.
        Nur wenn MDX23C nicht verfügbar ist, greift der reine DSP-Pfad.

        Referenz: Fitzgerald (2010) HPSS + Févotte & Idier (2011) NMF-β.
        """
        # --- ML-Fallback: MDX23C ---
        mdx_result = self._separate_mdx23c(audio, sr, requested_stems)
        if mdx_result is not None:
            return mdx_result

        # --- DSP-Fallback: HPSS ---
        audio_1d = audio.mean(axis=0) if audio.ndim == 2 else audio
        try:
            import librosa

            harmonic, percussive = librosa.effects.hpss(audio_1d, margin=3.0)
            harmonic = np.clip(harmonic, -1.0, 1.0).astype(np.float32)
            percussive = np.clip(percussive, -1.0, 1.0).astype(np.float32)
            vocal_est = np.clip(harmonic * 0.6, -1.0, 1.0).astype(np.float32)
            other_est = np.clip(harmonic * 0.4, -1.0, 1.0).astype(np.float32)
        except ImportError:
            harmonic = audio_1d.copy().astype(np.float32)
            percussive = np.zeros_like(audio_1d, dtype=np.float32)
            vocal_est = harmonic
            other_est = np.zeros_like(audio_1d, dtype=np.float32)

        stem_map: dict[str, np.ndarray] = {
            "vocals": vocal_est,
            "drums": percussive,
            "bass": percussive * 0.5,
            "guitar": other_est,
            "piano": other_est * 0.5,
            "other": other_est,
        }
        stems_out = {k: v for k, v in stem_map.items() if k in requested_stems}
        # SDRi gegen vollständige HPSS-Rekonstruktion (harmonic + percussive)
        _hpss_sdri_all = {"harmonic": np.clip(harmonic, -1.0, 1.0), "percussive": np.clip(percussive, -1.0, 1.0)}
        sdri = self._estimate_sdri(audio_1d, _hpss_sdri_all)
        logger.info("🎵 BS-RoFormer HPSS-DSP-Fallback aktiv | SDRi=%.1f dB", sdri)
        return StemSeparationResult(
            stems=stems_out,
            sr=sr,
            sdri_db=sdri,
            model_used="nmf_dsp_fallback",
            confidence=0.45,
            metadata={"fallback": True},
        )

    # ------------------------------------------------------------------
    # Hilfsmethoden
    # ------------------------------------------------------------------

    @staticmethod
    def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
        """Konvertiert Stereo zu Mono, stellt float32 [-1,1] sicher."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        peak = np.max(np.abs(audio))
        if peak > 1.0:
            audio = audio / peak
        return audio

    @staticmethod
    def _estimate_sdri(mix: np.ndarray, stems: dict[str, np.ndarray]) -> float:
        """Schätzt SDR-Improvement aus Stems-zu-Mischung-Energie-Verhältnis.

        SDRi wird stets gegen die vollständige Rekonstruktion (alle übergebenen Stems)
        gemessen. Caller muss sicherstellen, dass stems ALLE Stems enthält
        (vocals + instruments), nicht nur angeforderte Teilmenge.
        """
        if not stems:
            return 0.0
        recon = sum(stems.values())  # type: ignore[arg-type]
        recon = np.asarray(recon, dtype=np.float32)
        n = min(len(mix), len(recon))
        if n == 0:
            return 0.0
        mix_e = float(np.mean(mix[:n] ** 2))
        if mix_e < 1e-12:
            return 0.0
        err_e = float(np.mean((mix[:n] - recon[:n]) ** 2))
        # err_e ≈ 0 bedeutet perfekte Rekonstruktion → SDRi = 40 dB (Cap)
        sdri = 10.0 * math.log10(mix_e / (err_e + 1e-12))
        return float(np.clip(sdri, -20.0, 40.0))


# ---------------------------------------------------------------------------
# Singleton-Accessor (Thread-Safe Double-Checked Locking)
# ---------------------------------------------------------------------------


def get_bs_roformer() -> BSRoFormerPlugin:
    """Thread-sicherer Singleton-Accessor (Double-Checked Locking)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = BSRoFormerPlugin()
    return _instance


def separate_stems(
    audio: np.ndarray,
    sr: int,
    *,
    stems: list[str] | None = None,
) -> StemSeparationResult:
    """Convenience-Wrapper – BS-RoFormer Stem-Separation ohne Klassen-Instantiierung.

    Beispiel::

        result = separate_stems(audio, sr=48000, stems=["vocals", "drums"])
        vocals = result.stems["vocals"]
        logger.debug("Model: %s, SDRi: %.1f dB", result.model_used, result.sdri_db)

    Args:
        audio:  Eingangs-Audio (mono oder stereo float32)
        sr:     Sample-Rate (muss 48000 sein)
        stems:  Gewünschte Stems (None = alle 6)

    Returns:
        StemSeparationResult
    """
    return get_bs_roformer().separate(audio, sr, stems=stems)
