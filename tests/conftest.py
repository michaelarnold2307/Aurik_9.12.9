"""
Aurik 10.0.0 - Zentralisierte Test Fixtures
=========================================

Pytest fixtures für alle Aurik-Tests.
Utilities und Generators sind in test_utils.py definiert.

Author: Aurik Testing Team
Date: 14. Februar 2026
Version: 1.0.0 - INITIAL CONSOLIDATION
"""

import gc
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Add tests directory to path for test_utils import
tests_dir = os.path.dirname(os.path.abspath(__file__))
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# Add project root — benötigt für Modul-Level-Imports wie `from backend.core.X import ...`
# in Testdateien, die mit --import-mode=importlib gesammelt werden.
project_root = os.path.dirname(tests_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Add backend/ to path — ermöglicht `from backend.core.X import ...` (= backend/core/X.py)
# in Testdateien und im Produktionscode (z.B. backend/core/causal_defect_graph.py).
backend_dir = os.path.join(project_root, "backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Add src/ to path — enthält validate_musical_goals, orchestrator_and_cli, etc.
src_dir = os.path.join(os.path.dirname(tests_dir), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import utilities from test_utils
from test_utils import (
    MEDIUM_SPECIFIC_THRESHOLDS,
    generate_audio_by_quality,
    generate_medium_specific_audio,
)

_AURIK_COMPONENTS: dict[str, Any] = {}

# Import Aurik components
try:
    from backend.core.musical_goals.musical_goals_metrics import MusicalGoalsChecker as _MusicalGoalsChecker

    _AURIK_COMPONENTS["MusicalGoalsChecker"] = _MusicalGoalsChecker
    AURIK_COMPONENTS_AVAILABLE = True
except ImportError:
    AURIK_COMPONENTS_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# §CI-Minimal-Umgebung (ci-cross-platform.yml installiert nur numpy/scipy/
# soundfile/pytest/pytest-timeout/pyyaml): Testdateien mit weiteren Dritt-
# bibliotheks-Abhängigkeiten werden von der Collection ausgenommen, statt
# mit ModuleNotFoundError abzubrechen (Befund 2026-09-08: alle Cross-
# Platform-Jobs rot wegen Collection-Fehlern). In vollen Umgebungen laufen
# diese Dateien normal.
# ═══════════════════════════════════════════════════════════════════════════
import importlib.util as _importlib_util

_CI_MINIMAL_DEP_FILES: dict[str, tuple[str, ...]] = {
    "sklearn": ("tests/test_adaptive_chain_builder.py", "tests/test_ml_defect_detector.py"),
    "pydantic": ("tests/test_aesthetic_judgment.py", "tests/test_data_models.py", "tests/test_tape_types.py"),
    "hypothesis": ("tests/test_core_utils_hypothesis.py", "tests/unit/test_hypothesis_fuzzing.py"),
    "psutil": ("tests/test_performance_profiler.py",),
    "librosa": (
        "tests/test_phase_01_ml_hybrid.py",
        "tests/test_tonal_balance_restorer.py",
        "tests/test_transparent_dynamics.py",
        "tests/unit/test_adaptive_stft_consistency.py",
        "tests/unit/test_dsp_formant_shifter.py",
    ),
    "fastapi": ("tests/unit/test_batch_and_htdemucs_security.py",),
    "mutagen": ("tests/unit/test_exporter_dither.py",),
    "PyQt5": ("tests/unit/test_gui_transport_contracts.py", "tests/unit/test_help_system_errorsimplifier.py"),
    "requests": ("tests/unit/test_model_downloader.py",),
    "onnxruntime": (
        "tests/onnx_skip/test_plugin_manager.py",
        "tests/onnx_skip/test_onnx_advanced.py",
        "tests/onnx_skip/test_onnx_runtime.py",
    ),
}

collect_ignore = [
    _file
    for _dep, _files in _CI_MINIMAL_DEP_FILES.items()
    if _importlib_util.find_spec(_dep) is None
    for _file in _files
]


# ═══════════════════════════════════════════════════════════════════════════
# PYTEST FIXTURES
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def musical_goals_checker():
    """Fixture: MusicalGoalsChecker instance"""
    if not AURIK_COMPONENTS_AVAILABLE:
        pytest.skip("Aurik components not available")
    checker_cls = _AURIK_COMPONENTS.get("MusicalGoalsChecker")
    assert checker_cls is not None
    return checker_cls()


@pytest.fixture(scope="module", params=["PRISTINE", "EXCELLENT", "GOOD", "FAIR", "POOR", "VERY_POOR", "EXTREME"])
def audio_by_quality_level(request) -> tuple[np.ndarray, int, str]:
    """
    Parametrized Fixture: Audio for all 7 Material Quality levels.
    scope=module: pro Modul nur einmal generiert (7× statt N×7).

    Returns:
        Tuple of (audio, sr, quality_level)
    """
    quality_level = request.param
    sr = 48000
    audio = generate_audio_by_quality(quality_level, sr=sr, duration=0.25)
    return audio, sr, quality_level


@pytest.fixture(
    scope="module",
    params=[
        "VINYL_LP_STEREO",
        "CASSETTE_TYPE_I",
        "CD_STANDARD",
        "MP3_320",
        "SACD_DSD",
        "CYLINDER_EDISON",
        "TAPE_15IPS",
        "FM_STEREO",
    ],
)
def audio_by_medium_type_short(request) -> tuple[np.ndarray, int, str]:
    """
    Parametrized Fixture: Audio for major medium types (short list for quick tests).
    scope=module + 0.25s: butter/filtfilt auf 12k statt 96k Samples.

    Returns:
        Tuple of (audio, sr, medium_type)
    """
    medium_type = request.param
    sr = 48000
    audio = generate_medium_specific_audio(medium_type, sr=sr, duration=0.25)
    return audio, sr, medium_type


@pytest.fixture(scope="module", params=list(MEDIUM_SPECIFIC_THRESHOLDS.keys()))
def audio_by_medium_type_full(request) -> tuple[np.ndarray, int, str]:
    """
    Parametrized Fixture: Audio for ALL 30+ medium types.
    scope=module + 0.25s: butter/filtfilt auf 12k statt 96k Samples (8× schneller).

    Returns:
        Tuple of (audio, sr, medium_type)
    """
    medium_type = request.param
    sr = 48000
    audio = generate_medium_specific_audio(medium_type, sr=sr, duration=0.25)
    return audio, sr, medium_type


# ═══════════════════════════════════════════════════════════════════════════
# SESSION-SCOPED PERFORMANCE FIXTURES
# Werden pro Worker einmalig erstellt — vermeiden wiederholte np.sin()-Generierung
# ═══════════════════════════════════════════════════════════════════════════

_SESSION_SR = 44100
_SESSION_DURATION = 0.5  # Sekunden — kurz genug für schnelle Tests
# Optional full GC cadence for long local runs.
# Default 0 disables expensive full collections in per-test teardown,
# which can trigger flaky pytest-timeout failures in large suites.
_GC_FULL_INTERVAL = int(os.environ.get("AURIK_TEST_FULL_GC_INTERVAL", "0"))
_gc_teardown_counter = 0


@pytest.fixture(scope="session")
def session_sr() -> int:
    """Session-Fixture: Standard-Samplerate 44100 Hz"""
    return _SESSION_SR


@pytest.fixture(scope="session")
def session_mono(session_sr) -> np.ndarray:
    """
    Session-Fixture: Mono-440-Hz-Sinus, 0.5s@44100Hz, float32.
    Einmal pro Worker erstellt — für alle Unit-Tests wiederverwendbar.
    """
    n = int(session_sr * _SESSION_DURATION)
    t = np.linspace(0, _SESSION_DURATION, n, endpoint=False, dtype=np.float32)
    result: np.ndarray = np.asarray(0.5 * np.sin(2 * np.pi * 440 * t), dtype=np.float32)
    return result


@pytest.fixture(scope="session")
def session_stereo(session_mono) -> np.ndarray:
    """
    Session-Fixture: Stereo-Audio (2×N), L=440Hz, R=890Hz (leicht verschoben).
    Basiert auf session_mono für Cache-Effizienz.
    """
    n = session_mono.shape[0]
    t = np.linspace(0, _SESSION_DURATION, n, endpoint=False, dtype=np.float32)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    return np.stack([session_mono, right])


@pytest.fixture(scope="session")
def session_silence(session_sr) -> np.ndarray:
    """Session-Fixture: Stille (Nullvektor), 0.5s@44100Hz, float32."""
    n = int(session_sr * _SESSION_DURATION)
    return np.zeros(n, dtype=np.float32)


@pytest.fixture(scope="session")
def session_noise(session_sr) -> np.ndarray:
    """Session-Fixture: Weißes Rauschen, 0.5s@44100Hz, float32."""
    rng = np.random.default_rng(seed=42)
    n = int(session_sr * _SESSION_DURATION)
    return rng.standard_normal(n).astype(np.float32) * 0.1


# ═══════════════════════════════════════════════════════════════════════════
# SPEICHERSCHUTZ — verhindert VS Code Crash bei großen Testläufen
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(autouse=True)
def _gc_after_test():
    """Autouse-Fixture: Garbage Collection nach jedem Test.

    Verhindert Speicher-Akkumulation über 267+ Testdateien.
    numpy-Arrays und importierte Module werden freigegeben.
    Overhead: gen0 GC ~1–3 ms pro Test.
    Volles GC ist optional über AURIK_TEST_FULL_GC_INTERVAL steuerbar.
    """
    yield
    global _gc_teardown_counter
    _gc_teardown_counter += 1
    try:
        # Fast path: collect only generation 0 to avoid long teardown stalls.
        gc.collect(0)
        # Optional full GC for local diagnostics (disabled by default).
        if _GC_FULL_INTERVAL > 0 and (_gc_teardown_counter % _GC_FULL_INTERVAL) == 0:
            gc.collect()
    except (KeyboardInterrupt, SystemExit, Exception):
        pass


def pytest_sessionfinish(session, exitstatus):
    """Best-effort shutdown for background managers created during tests."""
    try:
        from backend.core import plugin_lifecycle_manager as _plm_mod

        mgr = vars(_plm_mod).get("_instance")
        if mgr is not None:
            mgr.shutdown()
        vars(_plm_mod)["_instance"] = None
    except (KeyboardInterrupt, SystemExit, Exception):
        pass


@pytest.fixture(scope="session")
def real_audio_gate_case() -> dict[str, object]:
    """Provide a short real-audio clip for acceptance tests (MP3-first).

    The fixture prefers local real-world media and returns a compact, centered
    clip to keep functional gate tests deterministic and runtime-bounded.
    """
    candidates = _real_audio_candidate_paths()
    preferred = [
        Path(project_root)
        / "test_audio"
        / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
        *candidates,
    ]
    audio_path = next((p for p in preferred if p.exists()), None)
    if audio_path is None:
        pytest.skip(
            "Keine reale Audio-Fixture gefunden. Erwartet eine Datei in: " + ", ".join(str(p) for p in preferred)
        )

    return _load_real_audio_clip(audio_path, max_seconds=8.0)


@pytest.fixture(scope="session")
def real_audio_corpus_cases() -> list[dict[str, object]]:
    """Provide a compact multi-file real-audio corpus for broad gates."""
    limit = max(1, int(float(os.environ.get("AURIK_REAL_AUDIO_CORPUS_LIMIT", "8") or 8)))
    cases: list[dict[str, object]] = []
    for audio_path in _real_audio_candidate_paths():
        if not audio_path.exists():
            continue
        try:
            cases.append(_load_real_audio_clip(audio_path, max_seconds=4.0))
        except pytest.skip.Exception:
            continue
        except Exception:
            continue
        if len(cases) >= limit:
            break
    if not cases:
        pytest.skip("Keine ladbaren Real-Audio-Korpusdateien gefunden.")
    return cases


def _real_audio_candidate_paths() -> list[Path]:
    root = Path(project_root)
    return [
        root / "test_audio" / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
        root / "test_audio" / "Elke Best - 30 Sekunden.mp3",
        root / "test_audio" / "tape" / "cassette_1980s_wow.wav",
        root / "test_audio" / "tape" / "reel_1940s_dropout.wav",
        root / "test_audio" / "vinyl" / "jazz_1950s_scratched.wav",
        root / "test_audio" / "vinyl" / "rock_1970s_worn.wav",
        root / "test_audio" / "digital" / "mp3_64kbps_artifacts.wav",
        root / "test_audio" / "digital" / "cd_clipped_2000s.wav",
        root / "test_audio" / "vocals" / "opera_sibilance.wav",
        root / "test_audio" / "vocals" / "choir_breaths.wav",
        root / "audio_examples" / "Elke Best - Du wolltest nur ein Abenteuer, aber ich suchte einen Freund.mp3",
        root / "audio_examples" / "Elke_Best_Freund.mp3",
        root / "temp_repro" / "repro_input.mp3",
    ]


def _load_real_audio_clip(audio_path: Path, *, max_seconds: float) -> dict[str, object]:
    from backend.file_import import load_audio_file

    loaded = load_audio_file(str(audio_path), target_sr=None, mono=False, do_carrier_analysis=False)
    if not loaded or loaded.get("audio") is None:
        pytest.fail(f"Reale Audio-Fixture konnte nicht geladen werden: {audio_path}")

    if loaded.get("error"):
        pytest.fail(f"Reale Audio-Fixture-Importfehler: {loaded['error']}")

    audio = np.asarray(loaded["audio"], dtype=np.float32)
    sr = int(loaded.get("sr") or 48_000)

    if audio.ndim == 2 and audio.shape[0] in (1, 2) and audio.shape[1] > audio.shape[0]:
        audio = audio.T

    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.ndim == 2 and audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)

    clip_len = min(audio.shape[0], int(sr * float(max_seconds)))
    if clip_len < int(sr * 2.0):
        pytest.skip(f"Reale Audio-Fixture ist zu kurz für Gate-Tests: {audio_path}")
    start = max(0, (audio.shape[0] - clip_len) // 2)
    clip = np.clip(np.nan_to_num(audio[start : start + clip_len], nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)

    # Resample only the short clip to avoid timeouts on long source files.
    if sr != 48_000:
        import librosa

        clip = librosa.resample(clip.T, orig_sr=sr, target_sr=48_000).T.astype(np.float32)
        clip = np.clip(np.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
        sr = 48_000

    return {
        "path": str(audio_path),
        "audio": clip,
        "sr": sr,
    }
