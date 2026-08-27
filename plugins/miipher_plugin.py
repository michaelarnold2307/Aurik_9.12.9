"""
MIIPHER Plugin — Vokal-Deep-Noise-Router (Legacy-Adapter, v10.0.0)

Das proprietäre Google-MIIPHER ist nicht gebündelt; dieses Modul ist der
kompatible Router für stark degradierten Gesang (SNR < 10 dB): die Route läuft
über SGMSE+ → DeepFilterNet v3.II → konservatives DSP mit expliziter
Route-Metadaten (model_used="miipher_deepfilternet_v3_ii").

Spec 04 Rev. 2026-08-15: Der SNR<10-Vokal-Task gehört SGMSE+ v2; Codec-
Degradation (mp3_low/streaming/aac/minidisc) trägt der offene DiT
(plugins/miipher_dit_plugin.py, Gate should_apply). Dieser Router bleibt als
Legacy-Pfad erhalten.

Model status: Native MIIPHER is not bundled. This module is still productive:
it routes deep-noise vocal material through the best local open-source chain
SGMSE+ → DeepFilterNet v3.II → conservative DSP, with explicit route metadata.

Aktivierung: Nur wenn DefectScanner `noise_snr_db < 10.0` UND
`panns_singing_confidence ≥ 0.35`.

§0h Invariante: Kein Output wird akzeptiert wenn artifact_freedom < 0.95.
§0j KI-Modell-Limitation: MIIPHER halluziniert potentiell Harmonics für
unbekannte Singstimmen → hallucination_guard.py nach Anwendung Pflicht.
"""

from __future__ import annotations

import logging
import os
import threading
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def _resolve_miipher_onnx_path() -> Path | None:
    """Löst den nativen MIIPHER-ONNX-Pfad auf (ENV-Override zuerst, dann Bundles)."""
    env_path = os.getenv("AURIK_MIIPHER_ONNX_PATH", "").strip()
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return candidate
        logger.debug("AURIK_MIIPHER_ONNX_PATH gesetzt aber Datei fehlt: %s", candidate)
        return None

    repo_root = Path(__file__).resolve().parents[1]
    candidates = (
        repo_root / "models" / "miipher" / "miipher.onnx",
        repo_root / "models" / "miipher.onnx",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


# Modell-Pfad (ENV-Override oder gebündelter Standardpfad)
_MIIPHER_ONNX_PATH = _resolve_miipher_onnx_path()

# SNR-Schwellwert für MIIPHER-Aktivierung (dB)
MIIPHER_SNR_THRESHOLD_DB = 10.0

# Minimum PANNs Gesangskonfidenz für MIIPHER
MIIPHER_SINGING_CONFIDENCE_MIN = 0.35

# DeepFilterNet Fallback energy_bias bei Gesang (§0j)
_DFN_FALLBACK_ENERGY_BIAS_DB = -6.0

# Native MIIPHER ONNX — Standard-SR für W2v-BERT-2.0 / HuBERT-basierte Modelle
_MIIPHER_NATIVE_SR_DEFAULT = 16000

# Singleton
_INSTANCE_HOLDER: list[MiipherPlugin | None] = [None]
_lock = threading.Lock()


def _load_symbol(module_name: str, symbol_name: str) -> Any:
    """Lädt an optional plugin/backend symbol lazily."""
    return getattr(import_module(module_name), symbol_name)


def _load_module(module_name: str) -> Any:
    """Lädt an optional module lazily."""
    return import_module(module_name)


class MiipherPlugin:
    """
    MIIPHER Last-Resort-Entrauscher für stark degradiertes Gesangsmaterial.

    Primary: MIIPHER ONNX (wenn Modell vorhanden).
    Fallback: DeepFilterNet v3.II mit Gesang-optimiertem energy_bias.

    Verwendung:
        plugin = get_miipher_plugin()
        if plugin.should_activate(noise_snr_db, panns_singing):
            result = plugin.enhance(audio, sr)
    """

    def __init__(self) -> None:
        self._model_loaded = False
        self._model_session = None
        self._last_miipher_path: str = "none"  # Tracking: welcher Pfad zuletzt genutzt wurde
        self._last_route_metadata: dict[str, object] = {
            "model_used": "none",
            "capability_status": "unavailable",
            "fallback_chain": [],
            "native_miipher_loaded": False,
            "activation_reason": "not_initialized",
        }
        self._try_load_model()

    @property
    def route_metadata(self) -> dict[str, object]:
        """Gibt metadata for the last enhancement route zurück."""
        return dict(self._last_route_metadata)

    def is_productive(self) -> bool:
        """True, wenn ein produktiver lokaler MIIPHER- oder Fallback-Pfad verfügbar ist."""
        if self._model_loaded and self._model_session is not None:
            return True
        try:
            sgmse_plugin = _load_module("plugins.sgmse_plugin")
            get_sgmse = getattr(sgmse_plugin, "get_loaded_sgmse_plus_plugin", None) or getattr(
                sgmse_plugin,
                "get_sgmse_plus_plugin",
                None,
            )
            sgmse = get_sgmse() if callable(get_sgmse) else None
            if sgmse is not None and bool(getattr(sgmse, "_model_loaded", False)):
                return True

            get_deepfilternet_plugin = _load_symbol("plugins.deepfilternet_v3_ii_plugin", "get_deepfilternet_plugin")
            dfn = get_deepfilternet_plugin()
            if dfn is not None and bool(getattr(dfn, "_model_loaded", False)):
                return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("MIIPHER adapter capability Pruefung nicht verfuegbar: %s", exc)
        return False

    def _try_load_model(self) -> None:
        """Try to load a native MIIPHER ONNX model if one is bundled."""
        if _MIIPHER_ONNX_PATH is None:
            logger.info(
                "MIIPHER native model not bundled — productive vocal SOTA adapter active "
                "(SGMSE+ -> DeepFilterNet v3.II -> DSP)."
            )
            return

        try:
            ort = _load_module("onnxruntime")
            get_ort_providers = _load_symbol("backend.core.ml_device_manager", "get_ort_providers")

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 4
            self._model_session = ort.InferenceSession(
                str(_MIIPHER_ONNX_PATH),
                sess_options=opts,
                providers=get_ort_providers("MIIPHER"),
            )
            self._model_loaded = True
            logger.info("MIIPHER ONNX geladen — §4.4 last-resort NR for SNR < 10 dB.")
            try:
                _reg = _load_symbol("backend.core.plugin_lifecycle_manager", "register_plugin")

                def _miipher_unload() -> None:
                    self._model_session = None
                    self._model_loaded = False

                _reg(
                    "MIIPHER",
                    size_gb=0.8,
                    unload_fn=_miipher_unload,
                )
            except Exception as _exc:
                logger.debug("PLM-Registrierung MIIPHER (unkritisch): %s", _exc)
        except Exception as exc:
            logger.debug("MIIPHER ONNX not loadable: %s — adapter Ersatzpfad active.", exc)

    def should_activate(self, noise_snr_db: float, panns_singing: float) -> bool:
        """
        Prüft ob MIIPHER für dieses Material sinnvoll ist.

        Args:
            noise_snr_db:   Geschätzter SNR in dB (aus DefectScanner)
            panns_singing:  PANNs Gesangskonfidenz [0,1]

        Returns:
            True wenn MIIPHER (oder sein Fallback) aktiviert werden soll.
        """
        return noise_snr_db < MIIPHER_SNR_THRESHOLD_DB and panns_singing >= MIIPHER_SINGING_CONFIDENCE_MIN

    def enhance(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        noise_snr_db: float = 0.0,
        vocal_energy_bias_db: float | None = None,  # §0p v10.0.0: register-adaptiver Bias aus VocalRegisterDetector
        panns_singing: float = 0.0,  # §0p v10.0.0: für SGMSE+ Vokal-Mode (konservativeres sigma)
    ) -> npt.NDArray[np.float32]:
        """
        Entrauscht stark degradiertes Gesangsmaterial.

        Primary: MIIPHER ONNX (wenn geladen).
        Fallback: DeepFilterNet v3.II (energy_bias register-adaptiv §0p).
        Last-Resort: Wiener-Filter als stets verfügbarer DSP-Fallback.

        §0h: artifact_freedom-Check NACH Anwendung in UV3 (nicht hier —
        Vermeidung von Doppel-Checks). Hier: nur NaN/Clip-Guard.

        Args:
            audio:               float32 Audio (mono/stereo, 48000 Hz)
            sr:                  Abtastrate (muss 48000 Hz sein)
            noise_snr_db:        Geschätzter Input-SNR (für Logging)
            vocal_energy_bias_db: Register-adaptiver energy_bias aus VocalRegisterDetector.
                                  None → SNR-adaptiver Default. Kopfstimme −3 dB, Brust −6 dB.

        Returns:
            Prozessiertes float32 Audio, gleiche Form wie Input.
        """
        assert sr == 48000, f"MIIPHER: SR muss 48000 Hz sein, erhalten: {sr}"

        reference = np.asarray(audio, dtype=np.float32)
        fallback_chain: list[str] = []
        self._last_route_metadata = {
            "model_used": "none",
            "capability_status": "unavailable",
            "fallback_chain": fallback_chain.copy(),
            "native_miipher_loaded": bool(self._model_loaded),
            "activation_reason": "miipher_entry",
        }

        # Productive open-source SOTA chain: SGMSE+ is the best available local
        # deep-noise vocal substitute for native MIIPHER when applied to a vocal stem.
        try:
            self._last_miipher_path = "none"
            result = self._enhance_miipher(reference, sr, panns_singing=float(panns_singing))  # §0p
            _is_native = self._last_miipher_path == "native_onnx"
            self._last_route_metadata = {
                "model_used": f"miipher_{self._last_miipher_path}",
                "capability_status": "sota_real" if _is_native else "sota_fallback",
                "fallback_chain": fallback_chain.copy(),
                "native_miipher_loaded": bool(self._model_loaded),
                "energy_bias_db": _DFN_FALLBACK_ENERGY_BIAS_DB,
                "activation_reason": "native_onnx" if _is_native else "sgmse_plus_chain",
            }
            return result
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            fallback_chain.append(f"sgmse_plus:{type(exc).__name__}")
            logger.debug("MIIPHER adapter SGMSE+ nicht verfuegbar: %s — DeepFilterNet Ersatzpfad.", exc)

        # Fallback: DeepFilterNet v3.II
        try:
            result = self._enhance_dfn_fallback(
                reference,
                sr,
                noise_snr_db=noise_snr_db,
                vocal_energy_bias_db=vocal_energy_bias_db,
                panns_singing=float(panns_singing),
            )
            _used_bias = vocal_energy_bias_db if vocal_energy_bias_db is not None else _DFN_FALLBACK_ENERGY_BIAS_DB
            self._last_route_metadata = {
                "model_used": "miipher_deepfilternet_v3_ii",
                "capability_status": "sota_fallback",
                "fallback_chain": fallback_chain.copy(),
                "native_miipher_loaded": bool(self._model_loaded),
                "energy_bias_db": _used_bias,
                "activation_reason": "deepfilternet_fallback",
            }
            return result
        except Exception as exc:  # pylint: disable=broad-except
            fallback_chain.append(f"deepfilternet_v3_ii:{type(exc).__name__}")
            logger.warning("DeepFilterNet Ersatzpfad fehlgeschlagen: %s — Wiener DSP Ersatzpfad.", exc)

        # Last-Resort: DSP Wiener-Filter
        result = self._enhance_wiener_fallback(reference, sr)
        self._last_route_metadata = {
            "model_used": "miipher_wiener_dsp",
            "capability_status": "dsp_fallback",
            "fallback_chain": fallback_chain.copy(),
            "native_miipher_loaded": bool(self._model_loaded),
            "energy_bias_db": _DFN_FALLBACK_ENERGY_BIAS_DB,
            "activation_reason": "wiener_last_resort",
        }
        return result

    def _run_native_miipher_onnx(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
    ) -> npt.NDArray[np.float32]:
        """Native MIIPHER ONNX Inferenz (§4.4 SOTA-Primary).

        Shape-adaptiv: [1,T] / [T] / [1,1,T] — erkennt Input-Nodes via get_inputs().
        Stereo: Pro-Kanal-Verarbeitung (L/R unabhängig — vermeidet Desync).
        Resampling: 48kHz → Modell-SR (ONNX-Metadaten oder 16kHz Default) → 48kHz.

        Raises:
            RuntimeError: wenn Inferenz fehlschlägt (→ Aufrufer fällt auf SGMSE+ zurück).
        """
        if self._model_session is None:
            raise RuntimeError("MIIPHER ONNX: _model_session ist None")

        # Input-Signatur lesen
        _inputs = self._model_session.get_inputs()
        if not _inputs:
            raise RuntimeError("MIIPHER ONNX: keine Input-Nodes gefunden")
        _input_name: str = _inputs[0].name
        _input_shape: list = list(_inputs[0].shape)

        # Modell-SR aus ONNX-Metadaten (custom_metadata_map) lesen
        try:
            _meta = self._model_session.get_modelmeta().custom_metadata_map
            _model_sr = int(_meta.get("sample_rate", _MIIPHER_NATIVE_SR_DEFAULT))
        except Exception:  # pylint: disable=broad-except
            _model_sr = _MIIPHER_NATIVE_SR_DEFAULT

        _is_stereo = audio.ndim == 2 and audio.shape[0] == 2
        _n_target = audio.shape[-1]

        def _process_channel(ch_audio: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
            """Resampled, führt ONNX-Inferenz durch und normalisiert Output für einen Mono-Kanal."""
            _librosa = _load_module("librosa")

            # 48kHz → model_sr
            _resampled: npt.NDArray[np.float32] = (
                _librosa.resample(ch_audio, orig_sr=sr, target_sr=_model_sr).astype(np.float32)
                if sr != _model_sr
                else ch_audio
            )

            # ONNX Input aufbereiten (Shape-adaptiv nach Anzahl Dimensionen)
            _ndim_exp = len(_input_shape)
            if _ndim_exp <= 1:
                _onnx_in = _resampled  # [T]
            elif _ndim_exp == 2:
                _onnx_in = _resampled[np.newaxis, :]  # [1, T]
            else:
                _onnx_in = _resampled[np.newaxis, np.newaxis, :]  # [1, 1, T]

            _onnx_out = self._model_session.run(None, {_input_name: _onnx_in})
            _raw = np.asarray(_onnx_out[0], dtype=np.float32)

            # Output-Shape normalisieren → [T] durch Squeeze führender Batch-Dimensionen
            while _raw.ndim > 1:
                _raw = _raw[0]

            # model_sr → 48kHz
            _enhanced: npt.NDArray[np.float32] = (
                _librosa.resample(_raw, orig_sr=_model_sr, target_sr=sr).astype(np.float32) if _model_sr != sr else _raw
            )

            # Länge deterministisch anpassen (kein Time-Warp)
            if _enhanced.shape[0] > _n_target:
                _enhanced = _enhanced[:_n_target]
            elif _enhanced.shape[0] < _n_target:
                _enhanced = np.pad(_enhanced, (0, _n_target - _enhanced.shape[0]), mode="edge")

            return np.clip(np.nan_to_num(_enhanced, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0).astype(np.float32)

        # Pro-Kanal-Verarbeitung (Mono und Stereo)
        if _is_stereo:
            _ch_l = _process_channel(audio[0].astype(np.float32))
            _ch_r = _process_channel(audio[1].astype(np.float32))
            _result: npt.NDArray[np.float32] = np.stack([_ch_l, _ch_r], axis=0).astype(np.float32)
        else:
            _result = _process_channel(audio.astype(np.float32))

        logger.info(
            "MIIPHER native ONNX: Inferenz erfolgreich (model_sr=%d, is_stereo=%s, n=%d)",
            _model_sr,
            _is_stereo,
            _n_target,
        )
        self._last_miipher_path = "native_onnx"
        return self._apply_vocal_safety_guards(audio, _result, sr, model_name="NativeMIIPHER")

    def _enhance_stem_sgmse(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        panns_singing: float = 0.0,
    ) -> npt.NDArray[np.float32]:
        """Stem-isoliertes SGMSE+ — Kern-Verbesserung gegenüber Full-Mix-Ansatz.

        Algorithmus (MIIPHER-Substitut, v10.0.0):
        1. MelBandRoFormer → vocal_stem (Mono-Soft-Mask aus Mono-Mixing)
        2. Stereo-Soft-Mask: vocal_stereo = audio × mask, instrumental = audio − vocal_stereo
        3. SGMSE+ nur auf vocal_stereo → keine Instrument-als-Rauschen-Konfusion
        4. Mix: enhanced_vocal + instrumental
        5. HNR-Blend + Hallucination-Guard (§0p/§2.46e)

        Qualitätsgates:
        - SDRi ≥ 1.0 dB (MBR-Ausgabe) → Separation hinreichend
        - Vocal-Stem-Energie ≥ −40 dBFS → überhaupt Inhalt

        Raises:
            RuntimeError: wenn MBR oder SGMSE+ nicht verfügbar oder Qualität unzureichend.
        """
        is_stereo = audio.ndim == 2 and audio.shape[0] == 2

        # --- 1. MelBandRoFormer vocal stem separation ---
        mbr_plugin_mod = _load_module("plugins.bs_roformer_plugin")
        get_mbr = getattr(mbr_plugin_mod, "get_bs_roformer_plugin", None)
        if get_mbr is None:
            raise RuntimeError("BS-RoFormer accessor unavailable for stem SGMSE+")
        mbr = get_mbr()
        if mbr is None:
            raise RuntimeError("BS-RoFormer plugin unavailable")

        sep_result = mbr.separate(audio, sr, stems=["vocals"])
        vocal_mono: npt.NDArray[np.float32] = np.asarray(
            sep_result.stems.get("vocals", np.zeros(audio.shape[-1], dtype=np.float32)),
            dtype=np.float32,
        )

        # SDRi-Qualitätsgate: nur fortfahren wenn Separation bedeutsam
        sdri = float(getattr(sep_result, "sdri", None) or 0.0)
        if sdri < 1.0:
            raise RuntimeError(f"BS-RoFormer SDRi={sdri:.2f} dB zu niedrig für Stem-SGMSE+ (min 1.0 dB)")

        # Vocal-Energie-Gate: stem muss hörbar sein
        _vocal_rms = float(np.sqrt(np.mean(vocal_mono**2)) + 1e-12)
        if 20.0 * np.log10(_vocal_rms) < -40.0:
            raise RuntimeError("Vocal-Stem-Energie < −40 dBFS — kein Gesang isolierbar")

        # --- 2. Stereo-Soft-Mask ---
        mix_mono: npt.NDArray[np.float32] = (
            audio.mean(axis=0).astype(np.float32) if is_stereo else audio.astype(np.float32)
        )
        # Soft Wiener-Mask: Verhältnis Vocal zu Mix (clamped [0,1])
        _abs_mix = np.abs(mix_mono)
        _abs_vocal = np.abs(vocal_mono)
        _mix_peak = float(np.percentile(_abs_mix, 99.9))
        if _mix_peak < 1e-6:
            raise RuntimeError("Mix-Peak zu gering — Stem-SGMSE+ nicht sinnvoll")
        mask: npt.NDArray[np.float32] = np.clip(_abs_vocal / (_abs_mix + 1e-7), 0.0, 1.0).astype(np.float32)

        if is_stereo:
            # Broadcast mask auf beide Kanäle
            vocal_stereo: npt.NDArray[np.float32] = (audio * mask[np.newaxis, :]).astype(np.float32)
        else:
            vocal_stereo = (audio * mask).astype(np.float32)

        instrumental: npt.NDArray[np.float32] = (audio - vocal_stereo).astype(np.float32)

        # --- 3. SGMSE+ nur auf Vocal-Stem ---
        sgmse_plugin = _load_module("plugins.sgmse_plugin")
        getter = (
            getattr(sgmse_plugin, "get_loaded_sgmse_plus_plugin", None)
            or getattr(sgmse_plugin, "get_sgmse_plus_plugin", None)
            or getattr(sgmse_plugin, "get_sgmse_plugin", None)
        )
        if getter is None:
            raise RuntimeError("SGMSE+ accessor unavailable")
        sgmse = getter()
        if sgmse is None or not bool(getattr(sgmse, "_model_loaded", False)):
            raise RuntimeError("SGMSE+ model not loaded")

        raw_vocal = sgmse.enhance(vocal_stereo, sr, panns_singing=float(panns_singing))
        raw_vocal_arr = getattr(raw_vocal, "audio", raw_vocal)
        enhanced_vocal: npt.NDArray[np.float32] = np.clip(
            np.nan_to_num(np.asarray(raw_vocal_arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )

        # Längenausgleich (SGMSE+ kann minimal kürzen)
        n_target = audio.shape[-1]
        if enhanced_vocal.shape[-1] < n_target:
            pad_len = n_target - enhanced_vocal.shape[-1]
            if is_stereo:
                enhanced_vocal = np.pad(enhanced_vocal, ((0, 0), (0, pad_len)), mode="edge")
            else:
                enhanced_vocal = np.pad(enhanced_vocal, (0, pad_len), mode="edge")
        elif enhanced_vocal.shape[-1] > n_target:
            if is_stereo:
                enhanced_vocal = enhanced_vocal[:, :n_target]
            else:
                enhanced_vocal = enhanced_vocal[:n_target]

        # --- 4. Mix: Enhanced-Vocal + Instrumental ---
        mixed: npt.NDArray[np.float32] = np.clip(
            np.nan_to_num(enhanced_vocal + instrumental, nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        ).astype(np.float32)

        # --- 5. Vocal Safety Guards (§0p/§2.46e) ---
        result = self._apply_vocal_safety_guards(audio, mixed, sr, model_name="StemSGMSE+")
        logger.debug(
            "MIIPHER adapter: Stem-SGMSE+ succeeded (SDRi=%.1f dB, is_stereo=%s)",
            sdri,
            is_stereo,
        )
        return result

    def _enhance_miipher(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        panns_singing: float = 0.0,  # §0p v10.0.0: weitergereicht an SGMSE+
    ) -> npt.NDArray[np.float32]:
        """MIIPHER-Kaskade: native ONNX → Stem-SGMSE+ → Full-Mix-SGMSE+.

        Prioritätskaskade:
        0. Native MIIPHER ONNX (§4.4 SOTA-Primary — nur wenn Modell geladen)
        1. Stem-SGMSE+ (MelBandRoFormer → SGMSE+ auf Vocal-Stem → Remix)
        2. Full-Mix-SGMSE+ (Fallback wenn MBR nicht verfügbar oder SDRi zu niedrig)

        Alle Pfade: HNR-Blend (§0p) + Hallucination-Guard (§2.46e).
        Raises RuntimeError → Aufrufer leitet an _enhance_dfn_fallback.
        """
        # Priorität 0: Native MIIPHER ONNX (§4.4 SOTA-Primary)
        if self._model_loaded and self._model_session is not None:
            try:
                return self._run_native_miipher_onnx(audio, sr)
            except Exception as _native_exc:  # pylint: disable=broad-except
                logger.debug(
                    "MIIPHER native ONNX Inferenz fehlgeschlagen: %s — SGMSE+ Ersatzpfad",
                    _native_exc,
                )

        # Priorität 1: Stem-based SGMSE+ (MIIPHER-Annäherung)
        try:
            _stem_result = self._enhance_stem_sgmse(audio, sr, panns_singing=float(panns_singing))
            self._last_miipher_path = "sgmse_plus_stem"
            return _stem_result
        except Exception as _stem_exc:
            logger.debug(
                "MIIPHER adapter: Stem-SGMSE+ nicht verfügbar (%s) — Full-Mix-SGMSE+ Ersatzpfad",
                _stem_exc,
            )

        # Priorität 2: Full-Mix-SGMSE+ (Fallback)
        try:
            sgmse_plugin = _load_module("plugins.sgmse_plugin")

            getter = (
                getattr(sgmse_plugin, "get_loaded_sgmse_plus_plugin", None)
                or getattr(sgmse_plugin, "get_sgmse_plus_plugin", None)
                or getattr(sgmse_plugin, "get_sgmse_plugin", None)
            )
            if getter is None:
                raise RuntimeError("SGMSE+ accessor unavailable")
            sgmse = getter()
            if sgmse is None or not bool(getattr(sgmse, "_model_loaded", False)):
                raise RuntimeError("SGMSE+ model not loaded")
            raw = sgmse.enhance(audio, sr, panns_singing=float(panns_singing))  # §0p Vokal-Mode
            raw_audio = getattr(raw, "audio", raw)
            enhanced: npt.NDArray[np.float32] = np.clip(
                np.nan_to_num(np.asarray(raw_audio, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            )
            enhanced = self._apply_vocal_safety_guards(audio, enhanced, sr, model_name="SGMSE+")

            logger.debug("MIIPHER adapter: SGMSE+ chain succeeded for deep-noise vocal restoration")
            self._last_miipher_path = "sgmse_plus_fullmix"
            return enhanced
        except RuntimeError:
            raise
        except Exception as exc:
            logger.debug("MIIPHER adapter: SGMSE+ error: %s — DFN Ersatzpfad", exc)
            raise RuntimeError(f"SGMSE+ not available for MIIPHER adapter: {exc}") from exc

    def _enhance_dfn_fallback(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,
        noise_snr_db: float = 0.0,
        vocal_energy_bias_db: float | None = None,  # §0p v10.0.0: register-adaptiv
        panns_singing: float = 0.0,  # §FCPE: Harmonik-Guard nur bei Gesang
    ) -> npt.NDArray[np.float32]:
        """DeepFilterNet v3.II Fallback mit register-adaptivem energy_bias (§0p).

        §9.12.9 Verbesserungen:
        - Register-adaptiver energy_bias: Kopfstimme −3 dB (hohe Harmonik-Dichte),
          Bruststimme −6 dB (Default), Fry/Flüstern −9 dB.
        - SNR-adaptiver Fallback: bei SNR < 5 dB weniger aggressiv (−4 dB)
          wenn kein Register-Bias übergeben wurde.
        - OMLSA Nachglättung: §SOTA-Matrix „DFN v3 + OMLSA".
        """
        get_deepfilternet_plugin = _load_symbol(
            "plugins.deepfilternet_v3_ii_plugin",
            "get_deepfilternet_plugin",
        )

        # §0p Register-adaptiver energy_bias hat Vorrang vor SNR-Schätzung.
        # Kein Register-Bias übergeben → SNR-adaptive Heuristik (Backwards-Compat).
        if vocal_energy_bias_db is not None:
            _snr_adaptive_bias = float(vocal_energy_bias_db)
        elif noise_snr_db < 5.0:
            _snr_adaptive_bias = -4.0  # milder für sehr tiefes SNR
        elif noise_snr_db > 8.0:
            _snr_adaptive_bias = -8.0  # aggressiver für mittleres SNR
        else:
            _snr_adaptive_bias = _DFN_FALLBACK_ENERGY_BIAS_DB

        dfn = get_deepfilternet_plugin()
        result_raw = dfn.enhance(audio, sr=sr, energy_bias_db=_snr_adaptive_bias)
        if result_raw is None or not np.isfinite(np.asarray(result_raw)).all():
            raise RuntimeError("DeepFilterNet-Fallback: ungültiges Ergebnis")

        out_f32: npt.NDArray[np.float32] = np.clip(np.asarray(result_raw, dtype=np.float32), -1.0, 1.0)

        # §9.12.8 OMLSA post-filter (§SOTA-Matrix „DFN v3 + OMLSA"): residual smoothing
        # nach DFN reduziert Musical Noise ohne Vokal-Timbral-Einbusse.
        # Implementierung: IMCRA-Rauschschätzung (compute_imcra_noise_estimate) +
        # spektraler Wiener-Gain (Ephraim-Malah Minimum MSE). energy_bias < 0 dB →
        # Rausch-PSD skalieren → Harmonik-Schutz (§DSP-Instructions §Noise-Schätzung).
        try:
            from scipy.signal import istft as _istft_om  # pylint: disable=import-outside-toplevel
            from scipy.signal import stft as _stft_om  # pylint: disable=import-outside-toplevel

            from backend.core.dsp.noise_estimator import (  # pylint: disable=import-outside-toplevel
                compute_imcra_noise_estimate as _compute_imcra_postfilter,
            )

            _n_fft_om = 2048
            _hop_om = 512  # 75 % Overlap (§STFT-Pflichtstandard)
            _omlsa_mono = out_f32.mean(axis=0) if out_f32.ndim == 2 else out_f32
            _noise_psd_om = _compute_imcra_postfilter(_omlsa_mono, sr, alpha_d=0.85, alpha_s=0.9)
            # energy_bias < 0 dB → Rausch-PSD reduzieren → Harmonik-Schutz
            _eb_lin = float(10.0 ** (float(_snr_adaptive_bias) * 0.5 / 10.0))
            _noise_psd_om = _noise_psd_om * max(_eb_lin, 1e-3)
            _chs_om = [out_f32[0], out_f32[1]] if out_f32.ndim == 2 and out_f32.shape[0] == 2 else [_omlsa_mono]
            _out_chs_om = []

            # §FCPE F0-guided harmonic protection (v10.0.0) — non-blocking.
            # Schützt Vokal-Harmoniken (F0·k, k=1..12) vor Over-Suppression im Wiener-Gain.
            # FCPE läuft auf DFN-Output (höherer SNR → zuverlässigere F0-Schätzung).
            _harmonic_gain_floor: np.ndarray | None = None
            try:
                if float(panns_singing or 0.0) >= 0.25:
                    _fcpe_plug = _load_symbol("plugins.fcpe_plugin", "get_fcpe_plugin")()
                    _fcpe_res = _fcpe_plug.analyze(_omlsa_mono, sr)
                    _f0_raw = np.asarray(_fcpe_res.f0_hz, dtype=np.float32)  # (T_fcpe,)
                    _n_fft_bins = _n_fft_om // 2 + 1
                    _freqs_om = np.linspace(0.0, sr / 2.0, _n_fft_bins, dtype=np.float32)
                    _n_frames_stft = max(1, int(len(_omlsa_mono) // _hop_om + 1))
                    _f0_interp = np.interp(
                        np.linspace(0, max(len(_f0_raw) - 1, 0), _n_frames_stft),
                        np.arange(len(_f0_raw)),
                        _f0_raw,
                    ).astype(np.float32)  # (n_frames_stft,)
                    # Harmonischen Schutz-Mask: ±40 Hz um jeden Oberton (F0·1..12)
                    _harm_bw_hz = 40.0
                    _hmask = np.zeros((_n_fft_bins, _n_frames_stft), dtype=np.float32)
                    for _k in range(1, 13):
                        _hfreq = _f0_interp[np.newaxis, :] * _k  # (1, n_frames)
                        _dist = np.abs(_freqs_om[:, np.newaxis] - _hfreq)  # (n_bins, n_frames)
                        _hmask += (_dist < _harm_bw_hz).astype(np.float32)
                    _hmask = np.clip(_hmask, 0.0, 1.0)
                    # Nur in voiced Frames (f0 > 0): Gain-Floor 0.80 → Harmoniken bleiben hörbar
                    _voiced_f = (_f0_interp > 0.0).astype(np.float32)[np.newaxis, :]
                    _harmonic_gain_floor = (_hmask * _voiced_f * 0.80).astype(np.float32)
                    logger.debug(
                        "MIIPHER DFN: FCPE F0-guided harmonic guard active (voiced=%d/%d frames)",
                        int(np.sum(_f0_interp > 0.0)),
                        _n_frames_stft,
                    )
            except Exception as _fcpe_exc:
                logger.debug("MIIPHER DFN: FCPE harmonic protection nicht blockierend: %s", _fcpe_exc)
                _harmonic_gain_floor = None

            for _ch_om in _chs_om:
                _, _, _Zxx = _stft_om(_ch_om, fs=sr, nperseg=_n_fft_om, noverlap=_n_fft_om - _hop_om, window="hann")
                _nf = min(_Zxx.shape[1], _noise_psd_om.shape[1])
                _spow = np.abs(_Zxx[:, :_nf]) ** 2
                _g_w = np.maximum(_spow - _noise_psd_om[:, :_nf], 0.0) / (_spow + 1e-12)
                # §FCPE Harmonik-Schutz: Minimum-Gain für Vokal-Harmoniken
                if _harmonic_gain_floor is not None:
                    _nf_h = min(_Zxx.shape[1], _harmonic_gain_floor.shape[1])
                    _g_w[:, :_nf_h] = np.maximum(_g_w[:, :_nf_h], _harmonic_gain_floor[:, :_nf_h])
                    # §v10.0.0 Interharmonic NR: cap Wiener gain in non-harmonic voiced bins
                    # → additional 6 dB suppression of residual noise between harmonics → improved HNR
                    _nonharm_cap = 1.0 - (1.0 - _hmask[:, :_nf_h]) * _voiced_f[:, :_nf_h] * 0.50
                    _g_w[:, :_nf_h] = np.minimum(_g_w[:, :_nf_h], _nonharm_cap)
                _Zxx_out = _Zxx.copy()
                _Zxx_out[:, :_nf] *= _g_w
                _, _ch_rec = _istft_om(_Zxx_out, fs=sr, nperseg=_n_fft_om, noverlap=_n_fft_om - _hop_om, window="hann")
                _out_chs_om.append(_ch_rec[: len(_ch_om)].astype(np.float32))
            _omlsa_f32: np.ndarray = (
                np.stack(_out_chs_om) if out_f32.ndim == 2 and out_f32.shape[0] == 2 else _out_chs_om[0]
            )
            _omlsa_f32 = np.clip(np.nan_to_num(_omlsa_f32, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
            if _omlsa_f32.shape == out_f32.shape:
                # Soft blend: 70% OMLSA result + 30% DFN (preserves DFN transient sharpness)
                out_f32 = np.clip(0.70 * _omlsa_f32 + 0.30 * out_f32, -1.0, 1.0)
                logger.debug("MIIPHER DFN: IMCRA post-filter angewendet (bias=%.1fdB)", _snr_adaptive_bias)
        except Exception as _omlsa_exc:
            logger.debug("MIIPHER DFN: OMLSA post-filter nicht blockierend: %s", _omlsa_exc)

        logger.debug(
            "MIIPHER adapter: DeepFilterNet Ersatzpfad succeeded (energy_bias=%.1f dB snr=%.1f dB)",
            _snr_adaptive_bias,
            noise_snr_db,
        )
        return self._apply_vocal_safety_guards(audio, out_f32, sr, model_name="DeepFilterNet")

    def _apply_vocal_safety_guards(
        self,
        audio_pre: npt.NDArray[np.float32],
        audio_post: npt.NDArray[np.float32],
        sr: int,
        *,
        model_name: str,
    ) -> npt.NDArray[np.float32]:
        """Wendet an: mandatory vocal NR guards after model inference."""
        enhanced = np.clip(
            np.nan_to_num(np.asarray(audio_post, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
            -1.0,
            1.0,
        )
        try:
            apply_hnr_blend = _load_symbol("backend.core.dsp.hnr_guard", "apply_hnr_blend")

            blended, hnr_diag = apply_hnr_blend(audio_pre, enhanced, sr)
            if float(hnr_diag.get("hnr_delta_db", 0.0)) > 3.0:
                logger.debug(
                    "MIIPHER adapter %s: HNR-Blend active (delta=%.1f dB)",
                    model_name,
                    float(hnr_diag.get("hnr_delta_db", 0.0)),
                )
            enhanced = np.clip(
                np.nan_to_num(np.asarray(blended, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0),
                -1.0,
                1.0,
            )
        except Exception as hnr_exc:  # pylint: disable=broad-except
            logger.debug("MIIPHER adapter %s: HNR-Blend nicht verfuegbar (nicht blockierend): %s", model_name, hnr_exc)

        try:
            check_hallucination = _load_symbol("backend.core.dsp.hallucination_guard", "check_hallucination")

            hg_result = check_hallucination(audio_pre, enhanced, sr=sr, mode="restoration")
            if getattr(hg_result, "requires_rollback", False):
                logger.info(
                    "MIIPHER adapter %s: Hallucination-Guard rollback (spectral_novelty=%.3f)",
                    model_name,
                    float(getattr(hg_result, "spectral_novelty", 0.0)),
                )
                raise RuntimeError(f"Hallucination-Guard rollback for {model_name}")
        except ImportError:
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
        except RuntimeError:
            raise
        except Exception as hg_exc:  # pylint: disable=broad-except
            logger.debug("MIIPHER adapter %s: Hallucination-Guard error (nicht blockierend): %s", model_name, hg_exc)

        return enhanced.astype(np.float32)

    def _enhance_wiener_fallback(
        self,
        audio: npt.NDArray[np.float32],
        sr: int,  # pylint: disable=unused-argument
    ) -> npt.NDArray[np.float32]:
        """
        Einfacher spektraler Wiener-Filter als Last-Resort DSP-Fallback.

        Schätzt Rauschspektrum aus lautestem 10%-Segment (umgekehrt: Signal dominiert dort),
        dann klassisches Wiener-Spektral-Subtraktionsfilter.
        """
        mono: np.ndarray
        is_stereo = audio.ndim == 2
        if is_stereo:
            if audio.shape[0] == 2 and audio.shape[1] > 2:
                mono = np.mean(audio, axis=0).astype(np.float64)
                is_channels_first = True
            else:
                mono = np.mean(audio, axis=1).astype(np.float64)
                is_channels_first = False
        else:
            mono = audio.astype(np.float64)
            is_channels_first = False

        n_fft = 2048
        hop = 512
        window = np.hanning(n_fft)

        frames = []
        for i in range(0, len(mono) - n_fft, hop):
            frame = mono[i : i + n_fft] * window
            frames.append(np.fft.rfft(frame))

        if not frames:
            return audio

        spectra = np.array(frames)  # (T, F)
        mag = np.abs(spectra)

        # Schätze Rauschprofil aus lautestem 10% der Frames invertiert
        # (im lauten Signal sieht man das Rausch-Minimum besser)
        frame_rms = np.sqrt(np.mean(mag**2, axis=1))
        quiet_thresh = np.percentile(frame_rms, 20)
        quiet_mask = frame_rms <= quiet_thresh
        if np.any(quiet_mask):
            noise_est = np.percentile(mag[quiet_mask], 50, axis=0)
        else:
            noise_est = np.percentile(mag, 10, axis=0)

        # Wiener-Gain G(f) = max(0.1, 1 - noise_est / (mag + ε))
        gain = np.clip(1.0 - noise_est[None, :] / np.clip(mag, 1e-10, None), 0.10, 1.0)
        spectra_filtered = spectra * gain

        # iSTFT via OLA
        out = np.zeros(len(mono))
        norm = np.zeros(len(mono))
        for i, (frame_filt, i_start) in enumerate(zip(spectra_filtered, range(0, len(mono) - n_fft, hop))):
            frame_t = np.fft.irfft(frame_filt).real[:n_fft] * window
            out[i_start : i_start + n_fft] += frame_t
            norm[i_start : i_start + n_fft] += window**2

        norm = np.where(norm > 1e-8, norm, 1.0)
        out /= norm
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        out = np.clip(out, -1.0, 1.0).astype(np.float32)

        logger.debug("MIIPHER: Wiener-Filter Last-Resort DSP-Ersatzpfad angewendet")

        if is_stereo:
            # Signal-Ratio Stereo-Rekonstruktion
            if is_channels_first:
                result = audio.copy()
                ratio = np.clip(out / (mono.astype(np.float32) + 1e-10), 0.0, 2.0)
                for ch in range(audio.shape[0]):
                    result[ch] = np.clip(audio[ch] * ratio, -1.0, 1.0)
            else:
                result = audio.copy()
                ratio = np.clip(out / (mono.astype(np.float32) + 1e-10), 0.0, 2.0)
                for ch in range(audio.shape[1]):
                    result[:, ch] = np.clip(audio[:, ch] * ratio, -1.0, 1.0)
            return result
        out_f32_w: npt.NDArray[np.float32] = out.astype(np.float32)
        return out_f32_w


def get_miipher_plugin() -> MiipherPlugin:
    """Thread-safe Singleton (Double-Checked Locking, §3.2)."""
    if _INSTANCE_HOLDER[0] is None:
        with _lock:
            if _INSTANCE_HOLDER[0] is None:
                _INSTANCE_HOLDER[0] = MiipherPlugin()
    plugin = _INSTANCE_HOLDER[0]
    assert plugin is not None
    return plugin
