"""
Smoke-Tests für alle 56 Aurik-Phasen — parametrisiert (§5.1 copilot-instructions.md).

Abgedeckt:      Phase 01–56 (alle Phasen-Klassen)
Ziel:           Einheitliche Grundprüfung, ersetzt 10 redundante test_phase_NN_*.py
                Einzeldateien (gelöscht 20.02.2026, Stufe 2 Bereinigung).

Pro Phase wird geprüft:
    - Import & Initialisierung fehlerfrei
    - process() gibt PhaseResult zurück mit success=True
    - Ausgang ist float-ndarray ohne NaN/Inf
    - Kein Hard-Clipping (|audio| ≤ 2.0)
    - Shape bleibt erhalten (Ausnahme: phase_32 Mono→Stereo)

Invarianten (§3.1 copilot-instructions.md):
    - np.random.seed(42) / default_rng(42) für Reproduzierbarkeit
    - Test < 30 s (pytest-timeout)
    - Nur synthetische Signale (keine realen Audio-Dateien)

Hinweis: Phase_51 hat zwei Klassen – DrumsEnhancementV1 wird getestet.
         Phase_32 ändert die Shape (mono→stereo), shape-Check entfällt dort.
"""

import contextlib
import importlib
import unittest.mock as _mock

import numpy as np
import pytest

from backend.core.defect_scanner import MaterialType

# MaterialType-Instanz für Phasen, die material als Pflicht-Arg benötigen
_MAT = MaterialType.CD_DIGITAL

# ─────────────────────────────────────────────────────────────────────────────
# Konstanten
# ─────────────────────────────────────────────────────────────────────────────
SR: int = 48_000
_N: int = SR // 10  # 4 800 Samples ≈ 0.1 s — kurz genug für 30 s-Timeout

_rng = np.random.default_rng(42)
_MONO: np.ndarray = np.clip(_rng.standard_normal(_N).astype(np.float32) * 0.30, -1.0, 1.0)
_STEREO: np.ndarray = np.clip(_rng.standard_normal((_N, 2)).astype(np.float32) * 0.30, -1.0, 1.0)
# Einige Phasen (z.B. phase_24_dropout_repair) benötigen längere Eingangssignale
# weil ihre internen Fensterbreiten > 4800 Samples sind.
_LONG_MONO: np.ndarray = np.clip(_rng.standard_normal(SR).astype(np.float32) * 0.30, -1.0, 1.0)
_LONG_STEREO: np.ndarray = np.clip(_rng.standard_normal((SR, 2)).astype(np.float32) * 0.30, -1.0, 1.0)
# Phasen, die ein 1-Sekunden-Signal benötigen (window_length > SR//10)
_NEEDS_LONG_AUDIO: frozenset = frozenset({"phase_24_dropout_repair"})

# Phasen mit schwerem ML-Singleton (LAION-CLAP, BEATs, SGMSE+) — try_allocate wird auf
# False gemockt, damit der DSP-Fallback greift und kein echtes Modell geladen wird (Unit-Test).
_HEAVY_ML_PHASES: frozenset = frozenset(
    {
        "phase_53_semantic_audio",
        "phase_49_advanced_dereverb",
        "phase_03_denoise",  # Resemble Enhance: nicht-deterministisch zwischen Aufrufen (§2.51)
        # §2.51: lädt BS-RoFormer/Demucs/PANNs/Whisper — echter ML-Pfad ist
        # nicht-deterministisch zwischen Aufrufen; Unit-Tests nutzen DSP-Fallback.
        "phase_42_vocal_enhancement",
    }
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase-Registry: (modul, klassenname, stereo_input, skip_shape_check)
#   stereo_input=True  → Phase erwartet zwingend Stereo
#   skip_shape_check   → Ausgangs-Shape muss nicht == Eingangs-Shape sein
# ─────────────────────────────────────────────────────────────────────────────
_PHASE_REGISTRY = [
    # modul                            klasse                          stereo  skip_shape
    ("phase_01_click_removal", "ClickRemovalPhase", False, False),
    ("phase_02_hum_removal", "HumRemovalPhase", False, False),
    ("phase_03_denoise", "DenoisePhase", False, False),
    ("phase_04_eq_correction", "EQCorrectionPhase", False, False),
    ("phase_05_rumble_filter", "RumbleFilterPhase", False, False),
    ("phase_06_frequency_restoration", "FrequencyRestorationPhase", False, False),
    ("phase_07_harmonic_restoration", "HarmonicRestorationPhase", False, False),
    ("phase_08_transient_preservation", "TransientPreservationPhase", False, False),
    ("phase_09_crackle_removal", "CrackleRemovalPhase", False, False),
    ("phase_10_compression", "CompressionPhase", False, False),
    ("phase_11_limiting", "LimitingPhase", False, False),
    ("phase_12_wow_flutter_fix", "WowFlutterFix", False, False),
    ("phase_13_stereo_enhancement", "StereoEnhancementPhaseV2", True, False),
    ("phase_14_phase_correction", "PhaseCorrection", True, False),
    ("phase_15_stereo_balance", "StereoBalancePhaseV2", True, False),
    ("phase_16_final_eq", "FinalEQ", False, False),
    ("phase_17_mastering_polish", "MasteringPolishPhase", False, False),
    ("phase_18_noise_gate", "NoiseGate", False, False),
    ("phase_19_de_esser", "DeEsserPhase", False, False),
    ("phase_20_reverb_reduction", "ReverbReduction", False, False),
    ("phase_21_exciter", "Exciter", False, False),
    ("phase_22_tape_saturation", "TapeSaturation", False, False),
    ("phase_23_spectral_repair", "SpectralRepair", False, False),
    ("phase_24_dropout_repair", "DropoutRepairPhase", False, False),
    ("phase_25_azimuth_correction", "AzimuthCorrectionPhaseV2", True, False),
    ("phase_26_dynamic_range_expansion", "DynamicRangeExpansion", False, False),
    ("phase_27_click_pop_removal", "ClickPopRemoval", False, False),
    ("phase_28_surface_noise_profiling", "SurfaceNoiseProfiling", False, False),
    ("phase_29_tape_hiss_reduction", "TapeHissReductionPhase", False, False),
    ("phase_30_dc_offset_removal", "DCOffsetRemoval", False, False),
    ("phase_31_speed_pitch_correction", "SpeedPitchCorrectionPhase", False, False),
    ("phase_32_mono_to_stereo", "MonoToStereoPhaseV2", False, True),  # mono→stereo
    ("phase_33_stereo_width_limiter", "StereoWidthLimiterPhaseV2", True, False),
    ("phase_34_mid_side_processing", "MidSideProcessing", True, False),
    ("phase_35_multiband_compression", "MultibandCompressionPhase", False, False),
    ("phase_36_transient_shaper", "TransientShaper", False, False),
    ("phase_37_bass_enhancement", "BassEnhancement", False, False),
    ("phase_38_presence_boost", "PresenceBoost", False, False),
    ("phase_39_air_band_enhancement", "AirBandEnhancement", False, False),
    ("phase_40_loudness_normalization", "LoudnessNormalizationPhase", False, False),
    ("phase_41_output_format_optimization", "OutputFormatOptimization", False, False),
    ("phase_42_vocal_enhancement", "VocalEnhancement", False, False),
    ("phase_43_ml_deesser", "MLDeEsserPhase", False, False),
    ("phase_44_guitar_enhancement", "GuitarEnhancementPhase", False, False),
    ("phase_45_brass_enhancement", "BrassEnhancementPhase", False, False),
    ("phase_46_spatial_enhancement", "SpatialEnhancementPhase", True, False),
    ("phase_47_truepeak_limiter", "TruePeakLimiterPhase", False, False),
    ("phase_48_stereo_width_enhancer", "StereoWidthEnhancerPhase", True, False),
    ("phase_49_advanced_dereverb", "AdvancedDereverbPhase", False, False),
    ("phase_50_spectral_repair", "SpectralRepairPhase", False, False),
    ("phase_51_drums_enhancement", "DrumsEnhancementV1", False, False),
    ("phase_52_piano_restoration", "PianoRestorationV1", False, False),
    ("phase_53_semantic_audio", "SemanticAudioPhase", False, False),
    ("phase_54_transparent_dynamics", "TransparentDynamicsV1", False, False),
    ("phase_55_diffusion_inpainting", "DiffusionInpaintingPhase", False, False),
    ("phase_56_spectral_band_gap_repair", "SpectralBandGapRepairPhase", False, False),
]

# Parametrisierungs-IDs = Modulnamen (gut lesbar im Pytest-Output)
_PARAMS = [pytest.param(mod, cls, stereo, skip_shape, id=mod) for mod, cls, stereo, skip_shape in _PHASE_REGISTRY]


# ─────────────────────────────────────────────────────────────────────────────
# Hilfsfunktion
# ─────────────────────────────────────────────────────────────────────────────


def _assert_phase_result(
    result: object,
    orig_audio: np.ndarray,
    phase_id: str,
    *,
    skip_shape_check: bool = False,
) -> None:
    """Prüft ein PhaseResult auf alle Grundinvarianten (§3.1 copilot-instructions.md).

    Args:
        result:           Rückgabe von phase.process()
        orig_audio:       Eingangs-Audio (für Shape-Vergleich)
        phase_id:         Modulname für aussagekräftige Fehlermeldungen
        skip_shape_check: True für Phasen die Shape absichtlich ändern (z.B. mono→stereo)
    """
    # Prüfe nach Klassenname statt isinstance() — importlib-Mode kann dieselbe
    # Klasse unter zwei Modulnamen registrieren (backend.core.* vs core.*),
    # was isinstance() trotz identischer Klasse fail lässt (§11.1 copilot-instructions.md).
    result_cls_name = type(result).__name__
    assert result_cls_name == "PhaseResult", (
        f"[{phase_id}] Kein PhaseResult: {type(result)} (Hinweis: importlib-Mode kann Doppelregistrierung verursachen)"
    )
    assert result.success is True, f"[{phase_id}] success=False — {result}"  # type: ignore[attr-defined]
    assert isinstance(result.audio, np.ndarray), f"[{phase_id}] result.audio ist kein ndarray"  # type: ignore[attr-defined]
    assert np.issubdtype(result.audio.dtype, np.floating), f"[{phase_id}] Dtype nicht float: {result.audio.dtype}"  # type: ignore[attr-defined]
    assert np.all(np.isfinite(result.audio)), (  # type: ignore[attr-defined]
        f"[{phase_id}] NaN/Inf im Ausgang: "
        f"nan={int(np.sum(np.isnan(result.audio)))} "  # type: ignore[attr-defined]
        f"inf={int(np.sum(np.isinf(result.audio)))}"  # type: ignore[attr-defined]
    )
    peak = float(np.max(np.abs(result.audio)))  # type: ignore[attr-defined]
    assert peak <= 2.0, f"[{phase_id}] Hard-Clipping: max|audio|={peak:.4f} > 2.0"
    if not skip_shape_check:
        assert result.audio.shape == orig_audio.shape, (  # type: ignore[attr-defined]
            f"[{phase_id}] Shape verändert: {orig_audio.shape} → {result.audio.shape}"  # type: ignore[attr-defined]
        )
    assert isinstance(result.metadata, dict), f"[{phase_id}] metadata kein dict"  # type: ignore[attr-defined]


# ─────────────────────────────────────────────────────────────────────────────
# Parametrisierter Smoke-Test
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mod_name,cls_name,use_stereo,skip_shape", _PARAMS)
def test_phase_smoke(mod_name: str, cls_name: str, use_stereo: bool, skip_shape: bool) -> None:
    """Smoke-Test für eine einzelne Phase.

    Parametrisiert über alle 56 Phasen. Prüft:
        1. Modul-Import und Klassen-Initialisierung
        2. process() mit dem korrekten Eingangs-Signal (mono/stereo)
        3. Alle Grundinvarianten via _assert_phase_result()

    Args:
        mod_name:   Modul-Name (z.B. 'phase_01_click_removal')
        cls_name:   Klassen-Name (z.B. 'ClickRemovalPhase')
        use_stereo: True, wenn die Phase Stereo-Eingang benötigt
        skip_shape: True, wenn Output-Shape von Input-Shape abweichen darf
    """
    # --- Import ---
    try:
        module = importlib.import_module(f"backend.core.phases.{mod_name}")
    except ImportError as exc:
        pytest.skip(f"Modul {mod_name} nicht importierbar: {exc}")

    phase_cls = getattr(module, cls_name, None)
    assert phase_cls is not None, f"Klasse '{cls_name}' nicht in backend.core.phases.{mod_name} gefunden"

    # --- Instanziierung ---
    try:
        phase = phase_cls()
    except Exception as exc:
        pytest.fail(f"[{mod_name}] Initialisierung fehlgeschlagen: {exc}")

    # --- Eingangs-Signal wählen ---
    if mod_name in _NEEDS_LONG_AUDIO:
        audio = _LONG_STEREO if use_stereo else _LONG_MONO
    else:
        audio = _STEREO if use_stereo else _MONO

    # --- Verarbeitung (material als Keyword → deckt auch Pflicht-Positional-Args ab) ---
    # Schwere ML-Phasen: try_allocate → False, damit DSP-Fallback greift (kein Modell-Load).
    _budget_patch = (
        _mock.patch("backend.core.ml_memory_budget.try_allocate", return_value=False)
        if mod_name in _HEAVY_ML_PHASES
        else contextlib.nullcontext()
    )
    try:
        with _budget_patch:
            result = phase.process(
                audio,
                sample_rate=SR,
                material=_MAT,  # Pflicht-Arg in phase_11/13/17/19/25/32/33/40/41
                material_type=_MAT,  # Alternative Benennung in einzelnen Phasen
            )
    except Exception as exc:
        pytest.fail(f"[{mod_name}] process() Exception: {exc}")

    # --- Invarianten prüfen ---
    _assert_phase_result(result, audio, mod_name, skip_shape_check=skip_shape)
