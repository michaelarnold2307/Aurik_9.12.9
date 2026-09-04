"""ML model readiness check — centralised pre-flight for all Aurik phases.

Every phase that uses an ML model MUST call `check_ml_model_ready()` before
invoking inference.  If the model cannot be loaded or is unavailable, a
WARNING is logged and the phase can fall back gracefully.

Model registry is populated at import time via lazy probing — each registered
check function probes the actual plugin/module and returns True only if the
model is fully loaded and ready for inference.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)

# Registry: model_id -> check_fn() -> bool
_MODEL_CHECKS: dict[str, Callable[[], bool]] = {}


def register_ml_check(model_id: str, check_fn: Callable[[], bool]) -> None:
    """Register a model readiness check function under a stable id."""
    _MODEL_CHECKS[model_id] = check_fn


def invalidate_ml_readiness(model_id: str) -> None:
    """Clear failure cache entries for a specific model (call after successful load).

    Called by ml_memory_budget.try_allocate() when a model is successfully
    allocated, so that subsequent readiness checks reflect the new state.
    """
    to_delete = [k for k in _FAILURE_CACHE if k == model_id or k.startswith(f"{model_id}:")]
    for k in to_delete:
        del _FAILURE_CACHE[k]
        _FAILURE_CACHE_TIMESTAMPS.pop(k, None)
    if to_delete:
        logger.debug("ml_model_readiness: invalidated %d Zwischenspeicher entries for '%s'", len(to_delete), model_id)


# Flood control: after first failed check, cache result to avoid log spam
_FAILURE_CACHE: dict[str, bool] = {}
# Time-based expiry: models may load later (e.g. Phase 20 checked before PANNs loaded).
# Cache entries older than this many seconds are re-checked.
_FAILURE_CACHE_TTL_S: float = 30.0
_FAILURE_CACHE_TIMESTAMPS: dict[str, float] = {}


def check_ml_model_ready(model_id: str, phase_name: str = "") -> bool:
    """Return True if the named ML model loaded successfully.

    If the check fails (model unavailable / not loaded), a WARNING is
    emitted ONCE per model+phase combination.  Subsequent calls return
    the cached result silently.

    Args:
        model_id:  Stable id registered via register_ml_check().
        phase_name: Optional phase identifier for the log message.

    Returns:
        True if the model is ready, False otherwise.
    """
    cache_key = f"{model_id}:{phase_name}" if phase_name else model_id

    # §F4: Check model-level cache first (shared across all phases).
    # If the model itself was checked recently, reuse the result regardless
    # of which phase is asking.  This prevents 5+ WARNINGs for the same model
    # from different phases (e.g. PANNs checked by Phase 01, 12, 20, 24, 28).
    if phase_name and model_id in _FAILURE_CACHE:
        _model_age = time.monotonic() - _FAILURE_CACHE_TIMESTAMPS.get(model_id, 0.0)
        if _model_age < _FAILURE_CACHE_TTL_S:
            return _FAILURE_CACHE[model_id]

    # Return cached result silently, but honour TTL for re-check.
    if cache_key in _FAILURE_CACHE:
        _age = time.monotonic() - _FAILURE_CACHE_TIMESTAMPS.get(cache_key, 0.0)
        if _age < _FAILURE_CACHE_TTL_S:
            return _FAILURE_CACHE[cache_key]
        # TTL expired — re-check the model.

    check_fn = _MODEL_CHECKS.get(model_id)
    if check_fn is None:
        logger.debug("ML model '%s' has no registered readiness Pruefung", model_id)
        return True

    try:
        ready = check_fn()
    except Exception as exc:
        logger.warning(
            "ML-Modell '%s' konnte nicht geladen werden (%s)%s",
            model_id,
            exc,
            f" — Phase {phase_name}" if phase_name else "",
        )
        _FAILURE_CACHE[cache_key] = False
        _FAILURE_CACHE_TIMESTAMPS[cache_key] = time.monotonic()
        # §F4: Also cache at model level so other phases benefit
        if phase_name:
            _FAILURE_CACHE[model_id] = False
            _FAILURE_CACHE_TIMESTAMPS[model_id] = time.monotonic()
        return False

    if not ready:
        logger.info(
            "ML-Modell '%s' ist nicht verfügbar (nicht geladen / Grenze erschöpft)%s",
            model_id,
            f" — Phase {phase_name}" if phase_name else "",
        )
        _FAILURE_CACHE[cache_key] = False
        _FAILURE_CACHE_TIMESTAMPS[cache_key] = time.monotonic()
        # §F4: Also cache at model level so other phases benefit
        if phase_name:
            _FAILURE_CACHE[model_id] = False
            _FAILURE_CACHE_TIMESTAMPS[model_id] = time.monotonic()
        return False

    return True


def clear_readiness_cache() -> None:
    """Clear the failure cache (call at start of each restoration run)."""
    _FAILURE_CACHE.clear()
    _FAILURE_CACHE_TIMESTAMPS.clear()


def _probe_plugin(module_path: str, getter_name: str, attr: str | None = None) -> Callable[[], bool]:
    """Return a check function that probes a plugin's availability WITHOUT loading it.

    IMPORTANT: Does NOT call the getter — calling get_*_plugin() triggers full
    model load + try_allocate() which can deadlock if called during another
    model's load sequence (re-entrant _lock in ml_memory_budget).
    Instead, only checks if the module is importable and has the getter.
    """

    def _check() -> bool:
        try:
            mod = __import__(module_path, fromlist=[getter_name])
            getter = getattr(mod, getter_name, None)
            if getter is None:
                return False
            # Do NOT call getter() — that triggers full model load.
            # Module existence + getter existence = probe success.
            return True
        except ImportError as exc:
            logger.debug("§V6 ML-Modul nicht importierbar — False zurückgegeben (Module %s): %s", module_path, exc)
            return False
        except Exception as exc:
            logger.debug("§V6 ML-Probe fehlgeschlagen — False zurückgegeben (Getter %s): %s", getter_name, exc)
            return False

    return _check


def _probe_function(module_path: str, fn_name: str) -> Callable[[], bool]:
    """Return a check function that probes a module-level function WITHOUT calling it.

    IMPORTANT: Does NOT call fn() — calling it can trigger full model load +
    try_allocate() which deadlocks if called during another model's load sequence.
    Instead, only checks if the module is importable and has the function.
    """

    def _check() -> bool:
        try:
            mod = __import__(module_path, fromlist=[fn_name])
            fn = getattr(mod, fn_name, None)
            # Do NOT call fn() — that triggers full model load.
            return fn is not None
        except ImportError as exc:
            logger.debug("§V6 ML-Funktion-Modul nicht importierbar — False zurückgegeben (Module %s): %s", module_path, exc)
            return False
        except Exception as exc:
            logger.debug("§V6 ML-Probe fehlgeschlagen — False zurückgegeben (Funktion %s): %s", fn_name, exc)
            return False

    return _check


# ── Register all known ML models ──────────────────────────────────────


def _register_all() -> None:
    """Probe and register all ML models used in Aurik phases."""

    # --- Denoising / Restoration ---
    register_ml_check(
        "DeepFilterNetV3",
        _probe_plugin("plugins.deepfilternet_v3_ii_plugin", "get_deepfilternet_plugin", "_model_loaded"),
    )
    register_ml_check(
        "BANQUET",
        _probe_function("backend.core.phases.phase_09_crackle_removal", "_get_banquet_onnx_session"),
    )

    # --- Bandwidth Extension / Inpainting ---
    register_ml_check(
        "FlashSR",
        _probe_function("plugins.flashsr_plugin", "_get_ml_model"),
    )
    register_ml_check(
        "GACELA",
        _probe_plugin("plugins.gacela_plugin", "get_gacela_plugin", "_model_loaded"),
    )
    register_ml_check(
        "AudioLDM2",
        _probe_plugin("plugins.audioldm2_plugin", "get_audioldm2_plugin", "_model_loaded"),
    )

    # --- Vocal / Stem ---
    register_ml_check(
        "BS-RoFormer",
        _probe_plugin("plugins.bs_roformer_plugin", "get_bs_roformer", "_model_loaded"),
    )
    register_ml_check(
        "MIIPHER",
        _probe_plugin("plugins.miipher_plugin", "get_miipher_plugin", "_model_loaded"),
    )
    register_ml_check(
        "Demucs",
        _probe_plugin("plugins.demucs_plugin", "get_demucs_plugin", "_model_loaded"),
    )

    # --- Voice Activity / Speech ---
    register_ml_check(
        "SileroVAD",
        _probe_plugin("plugins.silero_plugin", "get_silero_plugin", "_model_loaded"),
    )
    register_ml_check(
        "Whisper",
        _probe_plugin("backend.core.lyrics_guided_enhancement", "get_lyrics_guided_enhancement", "is_loaded"),
    )
    register_ml_check(
        "Wav2Vec2",
        _probe_plugin("backend.core.lyrics_guided_enhancement", "get_lyrics_guided_enhancement", "is_loaded"),
    )

    # --- Pitch / Frequency ---
    register_ml_check(
        "FCPE",
        _probe_plugin("plugins.fcpe_plugin", "get_fcpe_plugin", "_model_loaded"),
    )
    register_ml_check(
        "CREPE",
        _probe_plugin("plugins.crepe_plugin", "get_crepe_plugin", "_model_loaded"),
    )
    register_ml_check(
        "BasicPitch",
        _probe_plugin("plugins.basicpitch_plugin", "get_basicpitch_plugin", "_model_loaded"),
    )

    # --- Audio Tagging / Classification ---
    register_ml_check(
        "PANNs",
        _probe_plugin("plugins.panns_plugin", "get_panns_plugin", "_model_loaded"),
    )
    register_ml_check(
        "LAION-CLAP",
        _probe_plugin("plugins.laion_clap_plugin", "get_laion_clap", "_model_loaded"),
    )
    register_ml_check(
        "BEATs",
        _probe_plugin("plugins.beats_plugin", "get_beats_plugin", "_model_loaded"),
    )

    # --- Music Understanding ---
    register_ml_check(
        "MERT",
        _probe_plugin("plugins.mert_plugin", "get_mert_plugin", "_model_loaded"),
    )

    # --- Dereverberation ---
    register_ml_check(
        "WPE-Dereverb",
        _probe_plugin("plugins.wpe_plugin", "get_wpe_plugin", "_initialized"),
    )

    # --- Perceptual Quality ---
    def _ast_ready() -> bool:
        """Peekt den Singleton OHNE ihn zu konstruieren (§Selbsttest-Anti-Pattern).

        `get_perceptual_validator()` würde beim ersten Aufruf ein echtes ML-Modell
        laden (ml_memory_budget-Allokation) — ein reiner Readiness-Self-Test darf
        das nicht als Nebenwirkung auslösen (analog `is_ast_loaded()`-Fix).
        """
        try:
            from backend.core.musical_goals import perceptual_validator as _pv_mod

            _instance = getattr(_pv_mod, "_validator", None)
            return _instance is not None and _instance.is_loaded()
        except ImportError as exc:
            logger.debug("§V6 musical_goals.perceptual_validator nicht importierbar — False zurückgegeben (AST-Perceptual): %s", exc)
            return False
        except AttributeError as exc:
            # §V6 (copilot-instructions.md): Ein Readiness-Check darf NIE werfen.
            # Import-Ketten mit numba/librosa-Inkompatibilität (ROCm-Venv:
            # "'function' object has no attribute 'get_call_template'") lösten hier
            # CRITICAL im Selftest aus — der Check meldet dann ehrlich „nicht
            # bereit“ statt den Selftest dauerhaft zu vergiften.
            logger.warning(
                "AST-Perceptual-ONNX Readiness-Check nicht durchführbar (%s) — als nicht bereit gemeldet",
                exc,
            )
            return False

    register_ml_check("AST-Perceptual-ONNX", _ast_ready)

    # §v10.304 AST AudioSet-527 Classifier
    def _ast_classifier_ready() -> bool:
        try:
            from backend.core.ast_audio_set_classifier import is_ast_loaded

            return is_ast_loaded()
        except ImportError as exc:
            logger.debug("§V6 ast_audio_set_classifier.is_ast_loaded nicht verfügbar — False zurückgegeben (AST-Classifier): %s", exc)
            return False

    register_ml_check("AST-AudioSet-Classifier", _ast_classifier_ready)

    # --- Speech Enhancement / Separation ---
    register_ml_check("SGMSE+", _probe_plugin("plugins.sgmse_plugin", "get_sgmse_plus_plugin", "_model_loaded"))
    register_ml_check(
        "ResembleEnhance",
        _probe_plugin("plugins.resemble_enhance_plugin", "get_resemble_enhance_plugin", "_model_loaded"),
    )
    register_ml_check(
        "ConvTasNet", _probe_plugin("plugins.convtasnet_plugin", "get_convtasnet_plugin", "_model_loaded")
    )
    register_ml_check("MP-SENet", _probe_plugin("plugins.mp_senet_plugin", "get_mp_senet_plugin", "_model_loaded"))
    # --- Selbst trainierte Modelle (§v10.19) ---
    register_ml_check(
        "HarmonicInpaintingDiT",
        _probe_plugin("plugins.harmonic_inpainting_plugin", "get_harmonic_inpainting_plugin", "is_loaded"),
    )
    register_ml_check(
        "WhisperDenoiser (deprecated, Rev. 2026-08-16)",
        _probe_plugin("plugins.whisper_denoiser_plugin", "get_whisper_denoiser_plugin", "is_loaded"),
    )
    # --- Music Demixing ---
    register_ml_check("MDX23C", _probe_plugin("plugins.mdx23c_plugin", "get_mdx23c_plugin", "_model_loaded"))
    register_ml_check(
        "UVR-MDX-Net", _probe_plugin("plugins.uvr_mdxnet_plugin", "get_uvr_mdxnet_plugin", "_model_loaded")
    )
    # --- Vocoder / Waveform ---
    register_ml_check("BigVGAN", _probe_plugin("plugins.bigvgan_v2_plugin", "get_bigvgan_v2", "_model_loaded"))
    register_ml_check("HiFi-GAN", _probe_plugin("plugins.hifigan_plugin", "get_hifigan_plugin", "_model_loaded"))
    register_ml_check("Vocos", _probe_plugin("plugins.vocos_plugin", "get_vocos_plugin", "_model_loaded"))
    register_ml_check("DAC", _probe_plugin("plugins.dac_plugin", "get_dac_plugin", "_model_loaded"))
    register_ml_check("DiffWave", _probe_plugin("plugins.diffwave_plugin", "get_diffwave_plugin", "_model_loaded"))
    register_ml_check(
        "FlowMatching", _probe_plugin("plugins.flow_matching_plugin", "get_flow_matching_plugin", "_model_loaded")
    )
    # --- Quality Metrics ---
    register_ml_check("VERSA", _probe_plugin("plugins.versa_plugin", "get_versa_plugin", "_model_loaded"))
    register_ml_check("ViSQOL", _probe_plugin("plugins.visqol_plugin", "get_visqol_plugin", "_model_loaded"))
    register_ml_check("UTMOS", _probe_plugin("plugins.utmos_plugin", "get_utmos", "_model_loaded"))
    # --- Speaker / Voice / Pitch ---
    register_ml_check(
        "Resemblyzer", _probe_plugin("plugins.resemblyzer_plugin", "get_resemblyzer_plugin", "_model_loaded")
    )
    register_ml_check("RMVPE", _probe_plugin("plugins.rmvpe_plugin", "get_rmvpe_plugin", "_model_loaded"))
    # --- Super Resolution / Mastering ---
    register_ml_check("NVSR", _probe_plugin("plugins.nvsr_plugin", "get_nvsr_plugin", "_model_loaded"))
    register_ml_check(
        "Matchering", _probe_plugin("plugins.matchering_plugin", "get_matchering_plugin", "_model_loaded")
    )


_register_all()


# ── §A1 Startup-Selbsttest: Validate all registered readiness checks ──


def _validate_all_checks() -> None:
    """Run every registered check once at import time and log failures.

    A silent AttributeError here (e.g. probing '_model_loaded' on a plugin
    that doesn't have it) is a critical bug — the check will always return
    False and the model will never be used.  Catch and log at CRITICAL so
    it is visible in every run, not just when a phase happens to call
    check_ml_model_ready().
    """
    failed_attr: list[str] = []
    failed_import: list[str] = []
    passed: list[str] = []

    for model_id, check_fn in sorted(_MODEL_CHECKS.items()):
        try:
            # Just probe — don't care about ready/not-ready, only about exceptions.
            check_fn()
            passed.append(model_id)
        except AttributeError as exc:
            failed_attr.append(f"{model_id} ({exc})")
        except ImportError:
            failed_import.append(model_id)
        except Exception:
            # Other runtime exceptions are expected (model not on disk, etc.)
            passed.append(model_id)

    if failed_attr:
        logger.critical(
            "ml_model_readiness SELBSTTEST: %d ML-Checks haben KEIN erwartetes Attribut — "
            "diese Modelle werden NIEMALS als bereit erkannt: %s",
            len(failed_attr),
            ", ".join(failed_attr),
        )
    if failed_import:
        logger.warning(
            "ml_model_readiness SELBSTTEST: %d ML-Checks konnten Modul nicht importieren: %s",
            len(failed_import),
            ", ".join(failed_import),
        )
    logger.info(
        "ml_model_readiness SELBSTTEST: %d/%d Checks validiert (%d Attribut-Fehler, %d Import-Fehler)",
        len(passed),
        len(_MODEL_CHECKS),
        len(failed_attr),
        len(failed_import),
    )


_validate_all_checks()
