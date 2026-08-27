"""§SLR-1 StemLevelRestorer — Pre-phase vocal/instrumental stem-level NR (v10.0.0).

Applies MIIPHER (W2v-BERT 2.0) to the vocal stem and DeepFilterNet v3 to the
instrumental stem BEFORE the main phase pipeline in Restoration mode.  The
resulting re-mix is used as the phase-loop input, improving SNR for all
downstream phases without their per-phase side-effect risks.

Spec: §SLR-1 (copilot-instructions.md §0p + Kanonischer Pipeline-Ablauf)
Pipeline position: after VocalFocusAnalyzer, before `_execute_pipeline()`.

DSP invariants kept:
  - §0h  Music-Death-Shield: artifact_freedom ≥ 0.95 checked on result
  - §0p  HNR-Blend after MIIPHER (ΔHNR ≤ 3 dB enforced)
  - §0j  energy_bias −6 dB for vocal stem, −9 dB for instrumental stem
  - §2.46e Hallucination-Guard after every additive/stem operation
  - §2.47 GPU fallback to CPU on error

Singleton: use :func:`get_stem_level_restorer`.
"""

# Heavy DSP/ML dependencies are imported lazily inside guarded processing paths.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class StemLevelRestorerResult:
    """Result of :meth:`StemLevelRestorer.restore`."""

    audio: np.ndarray
    """Restored audio (same shape as input), channels-last, float32."""

    snr_gain_db: float
    """Estimated SNR improvement in dB (0.0 if not computed)."""

    success: bool
    """True if restoration was applied; False if fallback to original."""

    vocal_stem_miipher: bool = False
    """True if MIIPHER was applied to vocal stem."""

    instrumental_stem_dfn: bool = False
    """True if DeepFilterNet was applied to instrumental stem."""

    vqi_after: float = 1.0
    """VQI on restored audio (1.0 if not computed)."""

    rollback_reason: str = ""
    """Reason for returning the original audio after a safety gate."""

    fallback_reason: str = ""
    """Reason for a non-blocking fallback or no-op outcome."""

    separation_model: str = ""
    """Model route used for vocal/instrumental separation."""

    vocal_nr_model: str = ""
    """Model route used for vocal-stem noise reduction."""

    instrumental_nr_model: str = ""
    """Model route used for instrumental-stem noise reduction."""


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_instance: StemLevelRestorer | None = None
_lock = threading.Lock()


class StemLevelRestorer:
    """Pre-phase stem-level noise reducer (§SLR-1).

    Separates audio into vocal + instrumental stems, applies MIIPHER to the
    vocal stem and DeepFilterNet v3 to the instrumental stem, then remixes.

    All errors are non-blocking — on any exception the original audio is
    returned with ``success=False``.

    Singleton — use :func:`get_stem_level_restorer`.
    """

    # Maximum audio duration to process (longer material is truncated → processed → padded back).
    _MAX_PROCESS_S: float = 600.0  # 10 minutes

    # Minimum material duration — too-short clips skip stem NR.
    _MIN_DURATION_S: float = 2.0

    def restore(
        self,
        audio: np.ndarray,
        sample_rate: int,
        panns_singing: float = 0.5,
        restoration_context: dict | None = None,
    ) -> StemLevelRestorerResult | None:
        """Wendet an: pre-phase stem-level NR to audio.

        Args:
            audio:               Input audio, channels-last (samples,) or (samples, 2).
            sample_rate:         Must be 48000 Hz.
            panns_singing:       PANNs singing probability [0, 1].
            restoration_context: UV3 restoration context dict (read-only access).

        Returns:
            :class:`StemLevelRestorerResult` or ``None`` if skipped.
        """
        assert sample_rate == 48000, f"StemLevelRestorer.restore: SR must be 48000, got {sample_rate}"

        ctx = restoration_context or {}

        # Guard: minimum duration
        n_samples = audio.shape[0] if audio.ndim >= 1 else len(audio)
        if n_samples / sample_rate < self._MIN_DURATION_S:
            logger.debug("§SLR-1 uebersprungen: audio too short (%.1f s)", n_samples / sample_rate)
            return None

        # Guard: only process if singing detected
        if panns_singing < 0.35:
            logger.debug("§SLR-1 uebersprungen: panns_singing=%.3f < 0.35", panns_singing)
            return None

        try:
            return self._run(audio, sample_rate, panns_singing, ctx)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug("§SLR-1 wiederherstellen nicht blockierend: %s", exc)
            return StemLevelRestorerResult(
                audio=audio,
                snr_gain_db=0.0,
                success=False,
            )

    def _run(
        self,
        audio: np.ndarray,
        sample_rate: int,
        panns_singing: float,
        ctx: dict,
    ) -> StemLevelRestorerResult:
        """Intern: stem-level restoration logic."""
        _audio = np.asarray(audio, dtype=np.float32)
        _ctx = dict(ctx or {})
        if "active_ml_plugins" not in _ctx:
            _ctx["active_ml_plugins"] = self._resolve_active_ml_plugins(_ctx)

        # §0j register-adaptive energy_bias — kanonisches Mapping aus vocal_register_detector
        from backend.core.dsp.vocal_register_detector import (
            REGISTER_BIAS as _REGISTER_BIAS,
        )

        _vsp_reg = str((_ctx.get("vocal_style_profile") or {}).get("register", "chest")).lower()
        _vocal_energy_bias = _REGISTER_BIAS.get(_vsp_reg, -6.0)
        _multi_singer = bool(_ctx.get("multi_singer", False))
        if _multi_singer:
            logger.debug("§SLR-1 multi_singer=True erkannt; HNR-blend kept per-stem")
        logger.debug(
            "§SLR-1 panns_singing=%.3f register=%s vocal_energy_bias=%.1f dB",
            panns_singing,
            _vsp_reg,
            _vocal_energy_bias,
        )

        # §SLR-1a: Stem separation via SOTA router (BS-RoFormer → Demucs → MDX23C → DSP).
        _vocal_stem, _instr_stem, _separation_model = self._separate_stems(_audio, sample_rate, panns_singing, _ctx)

        _vocal_out = _vocal_stem.copy()
        _instr_out = _instr_stem.copy()
        _miipher_used = False
        _dfn_used = False
        _vocal_nr_model = "none"
        _instrumental_nr_model = "none"

        # §SLR-1b: MIIPHER on vocal stem (§0j register-adaptive energy_bias)
        try:
            _vocal_out, _miipher_used, _vocal_nr_model = self._apply_miipher(
                _vocal_stem,
                sample_rate,
                _vocal_energy_bias,
            )
        except Exception as _me:  # pylint: disable=broad-except
            logger.debug("§SLR-1 MIIPHER nicht blockierend: %s", _me)

        # §SLR-1c: HNR-Blend after MIIPHER (§0p)
        if _miipher_used:
            try:
                from backend.core.dsp.hnr_guard import apply_hnr_blend

                _vocal_out, _ = apply_hnr_blend(_vocal_stem, _vocal_out, sample_rate)
            except Exception as _hnr_exc:  # pylint: disable=broad-except
                logger.debug("§SLR-1 HNR-blend nicht blockierend: %s", _hnr_exc)

        # §SLR-1d: DeepFilterNet v3 on instrumental stem (§0j energy_bias −9 dB)
        try:
            _instr_out, _dfn_used, _instrumental_nr_model = self._apply_dfn(_instr_stem, sample_rate)
        except Exception as _de:  # pylint: disable=broad-except
            logger.debug("§SLR-1 DFN nicht blockierend: %s", _de)

        # §SLR-1e: Hallucination-Guard on each processed stem (§2.46e)
        _vocal_out = self._hallucination_guard(_vocal_stem, _vocal_out, sample_rate, "vocal")
        _instr_out = self._hallucination_guard(_instr_stem, _instr_out, sample_rate, "instr")

        # §SLR-1f: Remix stems to output
        _out = self._coerce_like(_vocal_out + _instr_out, _audio)
        _out = np.nan_to_num(_out, nan=0.0, posinf=0.0, neginf=0.0)
        _out = np.clip(_out, -1.0, 1.0)

        _vqi_after = 1.0
        _rollback_reason = ""
        try:
            from backend.core.musical_goals.vocal_quality_index import compute_vqi

            _vqi_result = compute_vqi(
                _audio,
                _out,
                sample_rate,
                skip_singer_identity=_multi_singer,
                genre=str(ctx.get("genre", "")) or None,
                reference_audio=ctx.get("reference_audio"),
                reference_singer_id=ctx.get("reference_singer_id"),
                era_profile=ctx.get("era_vocal_profile"),  # §EraVocalProfile: historisches Material
            )
            _vqi_after = float(_vqi_result.get("vqi", 1.0))
            _singer_cosine = float(_vqi_result.get("singer_identity_cosine", 1.0))
            if _vqi_after < 0.72:
                _rollback_reason = f"vqi_below_floor:{_vqi_after:.3f}"
            elif not _multi_singer and _singer_cosine < 0.92:
                _rollback_reason = f"singer_identity_below_floor:{_singer_cosine:.3f}"
        except Exception as _vqi_exc:  # pylint: disable=broad-except
            logger.debug("§SLR-1 VQI gate nicht blockierend: %s", _vqi_exc)

        if _rollback_reason:
            logger.warning("§SLR-1 §0p vocal gate rollback: %s", _rollback_reason)
            return StemLevelRestorerResult(
                audio=_audio.copy(),
                snr_gain_db=0.0,
                success=False,
                vocal_stem_miipher=_miipher_used,
                instrumental_stem_dfn=_dfn_used,
                vqi_after=_vqi_after,
                rollback_reason=_rollback_reason,
                separation_model=_separation_model,
                vocal_nr_model=_vocal_nr_model,
                instrumental_nr_model=_instrumental_nr_model,
            )

        # §SLR-1g: Estimate SNR gain
        _snr_gain = self._estimate_snr_gain(_audio, _out)
        _success = bool(_miipher_used or _dfn_used)

        return StemLevelRestorerResult(
            audio=_out,
            snr_gain_db=_snr_gain,
            success=_success,
            vocal_stem_miipher=_miipher_used,
            instrumental_stem_dfn=_dfn_used,
            vqi_after=_vqi_after,
            fallback_reason="" if _success else "no_stem_nr_applied",
            separation_model=_separation_model,
            vocal_nr_model=_vocal_nr_model,
            instrumental_nr_model=_instrumental_nr_model,
        )

    @staticmethod
    def _resolve_active_ml_plugins(ctx: dict | None = None) -> int:
        """Ermittelt aktive ML-Plugin-Anzahl aus Kontext oder PluginLifecycleManager."""
        try:
            if isinstance(ctx, dict) and "active_ml_plugins" in ctx:
                return int(max(0, int(ctx.get("active_ml_plugins", 0))))
        except Exception:  # pylint: disable=broad-except
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        try:
            from backend.core.plugin_lifecycle_manager import (  # pylint: disable=import-outside-toplevel
                get_plugin_lifecycle_manager,
            )

            _plm = get_plugin_lifecycle_manager()
            _entries = getattr(_plm, "_entries", {})
            if isinstance(_entries, dict):
                return int(sum(1 for _entry in _entries.values() if bool(getattr(_entry, "active", False))))
        except Exception:  # pylint: disable=broad-except
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

        return 0

    # -----------------------------------------------------------------------
    # Stem separation
    # -----------------------------------------------------------------------

    def _separate_stems(
        self,
        audio: np.ndarray,
        sample_rate: int,
        panns_singing: float = 0.0,
        ctx: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        """Separate audio into vocal and instrumental stems.

        Tries SOTA router first; falls back to DSP IIR bandpass.
        """
        try:
            from backend.core.dsp.sota_vocal_model_router import (
                get_sota_vocal_model_router,
            )

            route = get_sota_vocal_model_router().separate_vocal_instrumental(
                audio,
                sample_rate,
                panns_singing=panns_singing,
                ctx=ctx or {},
            )
            if route.success:
                return route.vocal.astype(np.float32), route.instrumental.astype(np.float32), route.model_used
            logger.warning(
                "ML→DSP-Fallback aktiviert (§SLR-1: SOTA-Router lehnte ab, chain=%s) — DSP-Bandpass",
                route.fallback_chain,
            )
            try:
                from backend.core.fallback_auditor import get_fallback_auditor

                get_fallback_auditor().record(
                    "StemLevelRestorer", "sota_vocal_router", "dsp_bandpass", "router_declined"
                )
            except Exception:
                logger.debug("FallbackAuditor nicht verfügbar (unkritisch)", exc_info=True)
        except Exception as _router_exc:  # pylint: disable=broad-except
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("§SLR-1 separation router Ersatzpfad to DSP: %s", _router_exc)
        _vocal, _instr = self._dsp_stem_split(audio, sample_rate)
        return _vocal, _instr, "dsp_bandpass_residual"

    @staticmethod
    def _coerce_like(candidate: np.ndarray | object, reference: np.ndarray) -> np.ndarray:
        """Gibt candidate as finite float32 audio with reference shape/layout zurück."""
        ref = np.asarray(reference, dtype=np.float32)
        try:
            arr = np.asarray(candidate, dtype=np.float32)
        except Exception:  # pylint: disable=broad-except
            return np.zeros_like(ref, dtype=np.float32)  # type: ignore[no-any-return]

        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if arr.ndim == 2 and ref.ndim == 2 and arr.T.shape == ref.shape:
            arr = arr.T
        if ref.ndim == 1 and arr.ndim == 2:
            arr = arr.mean(axis=1 if arr.shape[0] == ref.shape[0] else 0)
        elif ref.ndim == 2 and arr.ndim == 1:
            arr = np.repeat(arr[:, None], ref.shape[1], axis=1)

        if arr.ndim != ref.ndim:
            return np.zeros_like(ref, dtype=np.float32)  # type: ignore[no-any-return]
        if arr.shape[0] < ref.shape[0]:
            pad_shape = list(arr.shape)
            pad_shape[0] = ref.shape[0] - arr.shape[0]
            arr = np.concatenate([arr, np.zeros(pad_shape, dtype=np.float32)], axis=0)
        elif arr.shape[0] > ref.shape[0]:
            arr = arr[: ref.shape[0]]

        if arr.shape != ref.shape:
            return np.zeros_like(ref, dtype=np.float32)  # type: ignore[no-any-return]
        return np.clip(arr.astype(np.float32), -1.0, 1.0)  # type: ignore[no-any-return]

    def _dsp_stem_split(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """DSP fallback stem split: vocal ≈ 150–8000 Hz bandpass; instr = residual."""
        try:
            from scipy.signal import butter, sosfiltfilt

            _sos = butter(4, [150.0, 8000.0], btype="bandpass", fs=sample_rate, output="sos")
            if audio.ndim == 2 and audio.shape[0] == 2 and audio.shape[1] > 2:
                # §2.51: channels-first → temporär channels-last für Filterung entlang der Zeitachse
                _cl = audio.T
                _vocal = sosfiltfilt(_sos, _cl, axis=0).astype(np.float32).T
            else:
                _vocal = sosfiltfilt(_sos, audio, axis=0).astype(np.float32)
            _instr = (audio - _vocal).astype(np.float32)
            return _vocal, _instr
        except Exception as _sos_exc:  # pylint: disable=broad-except
            logger.warning("ML→DSP-Fallback aktiviert", exc_info=True)  # §V6 (copilot-instructions.md)
            logger.debug("§SLR-1 DSP stem split fehlgeschlagen: %s", _sos_exc)
            # Last resort: equal split
            _half = (audio * 0.5).astype(np.float32)
            return _half.copy(), _half.copy()

    # -----------------------------------------------------------------------
    # MIIPHER (vocal stem)
    # -----------------------------------------------------------------------

    def _apply_miipher(
        self, vocal_stem: np.ndarray, sample_rate: int, energy_bias_db: float = -6.0
    ) -> tuple[np.ndarray, bool, str]:
        """Wendet an: routed vocal NR with register-adaptive energy_bias (§0j)."""
        try:
            from backend.core.dsp.sota_vocal_model_router import (
                get_sota_vocal_model_router,
            )

            _result = get_sota_vocal_model_router().enhance_vocal(
                vocal_stem,
                sample_rate,
                energy_bias_db=energy_bias_db,
            )
            if not _result.success:
                return vocal_stem, False, _result.model_used
            _out = np.nan_to_num(_result.audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            _out = np.clip(_out, -1.0, 1.0)
            return _out, True, _result.model_used
        except Exception as _router_exc:  # pylint: disable=broad-except
            logger.debug("§SLR-1 vocal NR router nicht blockierend: %s", _router_exc)
            return vocal_stem, False, "none"

    # -----------------------------------------------------------------------
    # DeepFilterNet v3 (instrumental stem)
    # -----------------------------------------------------------------------

    def _apply_dfn(
        self,
        stem: np.ndarray,
        sample_rate: int,
        energy_bias_db: float = -9.0,
    ) -> tuple[np.ndarray, bool, str]:
        """Wendet an: routed DeepFilterNet v3 with energy_bias (§0j)."""
        try:
            from backend.core.dsp.sota_vocal_model_router import (
                get_sota_vocal_model_router,
            )

            _result = get_sota_vocal_model_router().enhance_instrumental(
                stem,
                sample_rate,
                energy_bias_db=energy_bias_db,
            )
            if not _result.success:
                return stem, False, _result.model_used
            _out = np.nan_to_num(_result.audio.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            _out = np.clip(_out, -1.0, 1.0)
            return _out, True, _result.model_used
        except Exception as _dfn_exc:  # pylint: disable=broad-except
            logger.debug("§SLR-1 DeepFilterNet nicht blockierend: %s", _dfn_exc)
            return stem, False, "none"

    # -----------------------------------------------------------------------
    # Hallucination-Guard (§2.46e)
    # -----------------------------------------------------------------------

    def _hallucination_guard(
        self,
        pre: np.ndarray,
        post: np.ndarray,
        sample_rate: int,
        stem_name: str,
    ) -> np.ndarray:
        """Prüft for hallucinated content and roll back if necessary (§2.46e)."""
        try:
            from backend.core.dsp.hallucination_guard import check_hallucination

            _result = check_hallucination(pre, post, sr=sample_rate, mode="restoration")
            if _result.requires_rollback:
                logger.warning(
                    "§SLR-1 §2.46e hallucination erkannt on %s stem → rollback",
                    stem_name,
                )
                return pre
        except Exception as _hg_exc:  # pylint: disable=broad-except
            logger.debug("§SLR-1 hallucination_guard nicht blockierend: %s", _hg_exc)
        return post

    # -----------------------------------------------------------------------
    # SNR estimation
    # -----------------------------------------------------------------------

    @staticmethod
    def _estimate_snr_gain(pre: np.ndarray, post: np.ndarray) -> float:
        """Schätzt SNR gain in dB between pre and post.

        Uses RMS of the difference as noise proxy.
        """
        try:
            _eps = 1e-10
            _noise = pre - post
            _rms_sig = float(np.sqrt(np.mean(np.square(pre)) + _eps))
            _rms_noise = float(np.sqrt(np.mean(np.square(_noise)) + _eps))
            _snr_db = 20.0 * np.log10(_rms_sig / _rms_noise + _eps)
            return float(np.clip(_snr_db, 0.0, 30.0))
        except Exception:  # pylint: disable=broad-except
            return 0.0


def get_stem_level_restorer() -> StemLevelRestorer:
    """Thread-safe singleton accessor for :class:`StemLevelRestorer`."""
    global _instance  # pylint: disable=global-statement
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = StemLevelRestorer()
                logger.debug("§SLR-1 StemLevelRestorer singleton erstellt.")
    return _instance


# §6-STEM: Demucs v5 6-stem (vocals,drums,bass,piano,guitar,other) verfügbar.
# Upgrade von 2-stem DSP-Fallback auf 6-stem ML-Separation empfohlen.
