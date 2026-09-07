"""
Root-Level conftest.py — wird von pytest VOR allen Sub-Verzeichnis-conftest.py-Dateien
geladen. Setzt BLAS-Thread-Limits bevor numpy importiert wird, um unter pytest-xdist
(parallele Worker-Prozesse) OpenBLAS-Thread-Oversubscription zu verhindern.

Problem ohne diesen Fix:
  8 xdist-Worker × OPENBLAS_NUM_THREADS=8 (Default) = 64 BLAS-Threads konkurrieren
  um dieselben CPU-Kerne → np.linalg.solve auf 256×256-Matrizen und np.roots auf
  Polynomen Grad 512 hängen länger als das 30s-pytest-Timeout.

Lösung:
  Jeder Worker nutzt genau 1 BLAS-Thread → keine gegenseitige Blockierung.
  Python-Level-Parallelismus (xdist-Worker-Prozesse) bleibt vollständig erhalten.

VS Code-Volltest-Stabilität (Anti-OOM + Anti-Pipe-Flood):
  URSACHE 1 — Pipe-Flood:
    VS Code vscode_pytest schickt fuer JEDEN der 5 200+ Tests eine eigene JSON-RPC-
    Nachricht ueber TEST_RUN_PIPE. Bei 5 200 Nachrichten laeuft der Extension-Host-
    Puffer voll → VS Code friert ein oder stuerzt ab.
    FIX: settings.json beschraenkt Test Explorer auf einen echten Smoke-Slice
    mit wenigen Dateien und weit unter der Host-Grenze. Vollsuite laeuft
    ausschliesslich ueber Terminal-Tasks (tasks.json).

  URSACHE 2 — ONNX-Singleton-OOM:
    Plugin-Singletons halten ONNX-Sessions (37-250 MB/Plugin). Nach ~267 Dateien
    akkumuliert sich >750 MB Singleton-RAM zusaetzlich zu VS Code (~1 GB).
    FIX (hier): _release_heavy_singletons() nach jedem Datei-Wechsel + GC.

  URSACHE 3 — Collection-Phase-Import-Kaskade:
    pytest laedt beim Sammeln ALLE 252 Test-Dateien (auch ohne sie auszufuehren).
    Module-Imports in Testdateien koennen ONNX-Sessions vorzeitig triggern.
    FIX (hier): pytest_collection_finish gibt Singletons direkt nach Collection frei.
"""

import gc as _gc
import logging
import os
import sys as _sys
import threading as _threading
import time as _time
import warnings as _warnings

import pytest

logger = logging.getLogger(__name__)

# Trio-Hinweis tritt in VS-Code-/pytest-Umgebungen mit eigenem excepthook auf
# und ist für unsere Testausführung nicht handlungsrelevant.
_warnings.filterwarnings(
    "ignore",
    message=r"You seem to already have a custom sys\.excepthook handler installed\..*",
    category=RuntimeWarning,
)

# Muss VOR dem ersten numpy-Import gesetzt werden.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TERM", "xterm-256color")

_VSCODE_LAST_FILE: str = ""
_VSCODE_TEST_COUNTER: int = 0
_VSCODE_GC_INTERVAL: int = 100
_VSCODE_FULL_GC_INTERVAL: int = int(os.environ.get("AURIK_TEST_FULL_GC_INTERVAL", "0"))

if any(_hint in " ".join(_sys.argv) for _hint in ("tests/integration", "tests/normative")):
    os.environ.setdefault("AURIK_SAFE_VALIDATION_PROFILE", "1")


def pytest_configure(config) -> None:
    """Wird im Haupt-Thread vor allen Tests aufgerufen.

    Setzt Drittanbieter-Warnungsfilter VOR der Collection.
    Diese müssen innerhalb von pytest_configure (nicht auf Modul-Ebene) gesetzt
    werden, damit sie NACH dem internen pytest -W-error-Hook laufen und thus an
    der vordersten Position in der Filter-Liste landen.
    """
    # pkg_resources-Deprecation kommt aus der Kette torch→resampy→pkg_resources.
    # Kein eigener Code betroffen — reine Drittanbieter-Warnung.
    _warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=UserWarning,
    )
    _warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API",
        category=DeprecationWarning,
    )

    # Guard 1: Wir sind bereits in einem xdist-Worker → kein Warm-up nötig.
    if os.environ.get("PYTEST_XDIST_WORKER"):
        return

    # Marker-Deklarationen für saubere Selektions-/Filter-UX.
    config.addinivalue_line("markers", "gui: tests that require Qt GUI runtime / display stack")

    # Integration-/Normative-Gates laufen in einem Safe-Validation-Profil,
    # damit teure Reporting-Nacharbeit und Neben-Threads die Gate-Laufzeit
    # nicht künstlich aufblasen.
    _pytest_targets = " ".join(str(arg) for arg in getattr(config, "args", []) or [])
    if any(_hint in _pytest_targets for _hint in ("tests/integration", "tests/normative")):
        os.environ.setdefault("AURIK_SAFE_VALIDATION_PROFILE", "1")

    # Guard 2: Kein xdist → kein Import-Lock-Risiko → Warm-up überspringen.
    # (Verhindert Numba-JIT-Hang bei pytest -p no:xdist und VS Code Test Explorer)
    numprocs = getattr(getattr(config, "option", None), "numprocesses", 0) or 0
    if not numprocs:
        return

    # Guard 3: xdist aktiv (numprocs > 0) → Warm-up im Haupt-Prozess.
    try:
        import librosa as _librosa
        import numpy as _np

        _d = _np.zeros(4096, dtype=_np.float32)
        _d[::4] = 0.1
        _librosa.stft(_d, n_fft=512, hop_length=128)
        _librosa.feature.mfcc(y=_d, sr=4000, n_mfcc=13)
        _librosa.feature.spectral_centroid(y=_d, sr=4000)
        _librosa.feature.spectral_rolloff(y=_d, sr=4000)
        _librosa.feature.chroma_stft(y=_d, sr=4000)
        _librosa.feature.rms(y=_d)
        _librosa.onset.onset_strength(y=_d, sr=4000)
        _librosa.beat.beat_track(y=_d, sr=4000)
        _ = _librosa.util.MAX_MEM_BLOCK
        _ = _librosa.util.frame
    except Exception:
        logger.warning("conftest.py::pytest_konfigurieren Ersatzpfad", exc_info=True)
        pass  # Kein Absturz — ist nur ein Warm-up


# Schwere ML-Plugin-Module mit ONNX-Sessions (je 37-250 MB RAM)
_HEAVY_PLUGIN_MODULES: tuple = (
    "plugins.deepfilternet_v3_ii_plugin",
    "plugins.apollo_plugin",
    "plugins.vocos_plugin",
    "plugins.crepe_plugin",
    "plugins.banquet_vinyl_plugin",
    "plugins.resemble_enhance_plugin",
    "plugins.panns_plugin",
    "plugins.hifigan_plugin",
    "plugins.uvr_mdxnet_plugin",
    "plugins.mp_senet_plugin",  # §4.4: MP-SENet 2023 (ersetzt DCCRN + FullSubNet+)
    "plugins.versa_plugin",  # §4.4: VERSA 2024 MOS (ersetzt CDPAM)
    "plugins.diffwave_plugin",
    "plugins.bs_roformer_plugin",
    "plugins.mdx23c_plugin",
    "plugins.demucs_v4_plugin",
    "plugins.wpe_plugin",
    "plugins.laion_clap_plugin",
    "plugins.flow_matching_plugin",
    "plugins.cqtdiff_plus_plugin",
)

_VSCODE_SAFE_TEST_LIMIT: int = int(os.environ.get("AURIK_VSCODE_TEST_LIMIT", "250"))
# Maximale Anzahl Test-Dateien im VS Code Test Explorer (verhindert Import-Kaskade + Pipe-Flood).
# Nur TEST_RUN_PIPE-Runs betroffen (exklusiv Test Explorer, nicht Terminal-Tasks).
_VSCODE_FILE_LIMIT: int = int(os.environ.get("AURIK_VSCODE_FILE_LIMIT", "50"))
_vscode_file_count: int = 0  # Zählt Test-Dateien während Collection in VS Code Test Explorer Runs


def _release_heavy_singletons() -> None:
    """Setzt _instance in allen schweren Plugin-Modulen auf None."""
    for mod_name in _HEAVY_PLUGIN_MODULES:
        mod = _sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "_instance"):
            try:
                mod._instance = None  # type: ignore[attr-defined]
            except Exception:
                logger.warning("conftest.py::_release_heavy_singletons Ersatzpfad", exc_info=True)


def _is_vscode_run() -> bool:
    """True wenn pytest vom VS Code Test Explorer (TEST_RUN_PIPE) gestartet wurde.

    VSCODE_PID wird in ALLEN VS Code Terminals gesetzt — auch in tasks.json-Runs, die
    die Vollsuite ausführen sollen. Nur TEST_RUN_PIPE ist exklusiv für den Test Explorer
    (vscode_pytest setzt diese Pipe für JSON-RPC-Streaming). Würde VSCODE_PID mitgeprüft,
    wären auch Terminal-Tasks auf 250 Tests begrenzt → falsches Verhalten.
    """
    return "TEST_RUN_PIPE" in os.environ


def pytest_collection_finish(session) -> None:
    """Nach vollstaendiger Test-Collection: Singletons freigeben + Warnung.

    Collection importiert alle Test-Module, was ONNX-Sessions vorzeitig
    triggern kann. Einmalig freigeben bevor Ausfuehrung startet.
    Warnt bei VS Code-Laeufen mit > 1 500 Tests.
    """
    n_tests = len(session.items)

    if _is_vscode_run() and n_tests > _VSCODE_SAFE_TEST_LIMIT:
        _warnings.warn(
            f"\n\nAURIK WARNUNG: {n_tests:,} Tests im VS Code Test Explorer.\n"
            f"Bei > {_VSCODE_SAFE_TEST_LIMIT:,} Tests droht VS Code-Absturz (Pipe-Flood).\n"
            f"Aurik kappt VS-Code-Laeufe deshalb automatisch auf diese Grenze.\n"
            f"Vollsuite: Terminal → Strg+Shift+B → 'Vollsuite parallel'\n",
            stacklevel=1,
            category=UserWarning,
        )

    # Die Integrations-/Normative-Suite soll als Safe-Validation laufen.
    # Collection kennt die echten Test-Dateien bereits, daher ist dies der
    # robusteste Zeitpunkt, um teure Post-Processing-Pfade für diese Gates
    # global zu deaktivieren.
    if any(
        "tests/integration/" in str(getattr(item, "fspath", ""))
        or "tests/normative/" in str(getattr(item, "fspath", ""))
        for item in session.items
    ):
        os.environ.setdefault("AURIK_SAFE_VALIDATION_PROFILE", "1")

    _release_heavy_singletons()
    # Fast incremental GC to avoid long collection stalls in VS Code.
    _gc.collect(0)


def pytest_runtest_teardown(item, nextitem) -> None:
    """GC und Singleton-Freigabe: nach Datei-Wechsel und alle 100 Tests.

    Verhindert den VS Code-OOM-Crash bei ~97 % der 5236-Test-Suite.
    """
    global _VSCODE_LAST_FILE, _VSCODE_TEST_COUNTER

    _VSCODE_TEST_COUNTER += 1
    current_file = str(getattr(item, "fspath", ""))

    if current_file and current_file != _VSCODE_LAST_FILE:
        _VSCODE_LAST_FILE = current_file
        _release_heavy_singletons()
        _gc.collect(0)
    elif _VSCODE_TEST_COUNTER % _VSCODE_GC_INTERVAL == 0:
        _gc.collect(0)

    if _VSCODE_FULL_GC_INTERVAL > 0 and _VSCODE_TEST_COUNTER % _VSCODE_FULL_GC_INTERVAL == 0:
        _gc.collect()


def pytest_sessionfinish(session, exitstatus) -> None:
    """Robuster End-of-Session-Cleanup für Hintergrund-Threads und ML-Singletons.

    Verhindert native Aborts beim Interpreter-Shutdown (z. B. Exit 134), wenn
    Monitor-Threads aus ARM/PLM nach Testende noch aktiv sind.
    """
    # 1) ARM-Monitor sicher stoppen (falls während Integration/E2E aktiviert).
    try:
        from backend.core.adaptive_resource_manager import (
            adaptive_resource_manager as _arm,
        )

        _arm.stop_monitoring()
    except Exception:
        logger.warning("conftest.py::pytest_sessionfinish Ersatzpfad", exc_info=True)

    # 2) PLM-Monitor sicher stoppen + Pipeline-Refcount defensiv entspannen.
    try:
        from backend.core.plugin_lifecycle_manager import (
            get_plugin_lifecycle_manager as _get_plm,
        )
        from backend.core.plugin_lifecycle_manager import (
            set_pipeline_active as _set_pipeline_active,
        )

        for _ in range(8):
            try:
                _set_pipeline_active(False)
            except Exception:
                break
        _get_plm().shutdown()
    except Exception:
        logger.warning("conftest.py::pytest_sessionfinish Ersatzpfad", exc_info=True)

    # 3) Schwere Modulsingletons freigeben + finales GC.
    _release_heavy_singletons()
    _gc.collect()

    # 4) AurikDenker-Restaurier-Threads koennen nach Timeout/Fallback noch kurz
    # weiterlaufen. Vor dem Interpreter-Shutdown geben wir ihnen einen kleinen,
    # deterministischen Drain-Puffer, damit kein nativer Abort bei noch aktiven
    # Daemon-Threads entsteht.
    _drain_deadline = _time.monotonic() + 30.0
    while _time.monotonic() < _drain_deadline:
        _rest_threads = []
        for _th in list(_threading.enumerate()):
            try:
                if _th is _threading.current_thread():
                    continue
                if not _th.is_alive():
                    continue
                if _th.name.endswith("(_run_rest)"):
                    _rest_threads.append(_th)
            except Exception:
                logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)
                continue

        if not _rest_threads:
            break

        _remaining = max(0.0, _drain_deadline - _time.monotonic())
        _slice = min(5.0, _remaining)
        if _slice <= 0.0:
            break

        for _th in _rest_threads:
            try:
                _th.join(timeout=_slice)
            except Exception:
                logger.warning("conftest.py::pytest_sessionfinish Ersatzpfad", exc_info=True)

    if os.environ.get("AURIK_PYTEST_THREAD_DUMP", "0") == "1":
        try:
            _alive = []
            for _th in _threading.enumerate():
                _alive.append(
                    {
                        "name": _th.name,
                        "daemon": bool(_th.daemon),
                        "alive": bool(_th.is_alive()),
                    }
                )
            print(f"AURIK_THREAD_DUMP sessionfinish exitstatus={exitstatus} threads={_alive}", file=_sys.stderr)
        except Exception:
            logger.warning("conftest.py::unknown Ersatzpfad", exc_info=True)


# ── Legacy-Testdateien ausschließen ────────────────────────────────────────
# Diese Dateien sind Script-Style-Tests oder testen Module, die nicht mehr
# existieren. Sie werden aus der pytest-Collection ausgeschlossen um
# Collection-Errors zu vermeiden. Neue Unit-Tests gehören nach tests/unit/.
collect_ignore: list[str] = [
    # Script-style Dateien (kein pytest-konformes Test-Layout, Modul-Level-Code):
    # ONNX-Tests benötigen torch (OSError: libcupti.so.12 in dieser venv):
    "tests/onnx/test_onnx_advanced.py",
    "tests/onnx/test_onnx_runtime.py",
]

collect_ignore.extend(
    [
        # Standalone stress script: not suitable for default pytest runs.
        "tests/memory_leak_test.py",
    ]
)

# Normalize legacy ignore file names for robust early collection filtering.
_LEGACY_IGNORE_BASENAMES: set[str] = {p.replace("\\", "/").split("/")[-1].lower() for p in collect_ignore}


_HEAVY_TEST_PATH_HINTS: tuple[str, ...] = (
    "test_defect_scanner_long_audio_crop_rescue.py",
    "test_memory_leaks_v3.py",
    "test_full_chain_ml_hybrid.py",
    "test_e2e_v9_10_41.py",
    "test_tier1_integration.py",
    "test_phase_skipping_integration.py",
    "test_phase_selection_complete.py",
    "test_ml_hybrid_integration.py",
    "test_panns_integration.py",
    "test_v99_sota_plugins.py",
    "test_v99_plugins_extended.py",
    "test_v99_vocos_plugin.py",
    "test_ml_era_detector.py",
    "test_ml_medium_detector.py",
    "test_ml_hybrid_validation.py",
    "test_demucs_v4_plugin.py",
    "test_gaps_integration.py",
    "test_digital_restoration_specialist.py",
    "test_phase31_ml_integration.py",
    "test_signal_forensics_integration.py",
    "test_unified_analyzer.py",
    "test_phase_23_ml_hybrid.py",
    "test_phase23_ml_hybrid.py",
)


def pytest_addoption(parser) -> None:
    """Add explicit switch to run heavy tests on demand.

    Default behaviour is crash-safe: heavy tests are skipped unless requested.
    """
    parser.addoption(
        "--run-heavy-tests",
        action="store_true",
        default=False,
        help="Run heavy ML/stress tests that are skipped by default for system stability.",
    )
    parser.addoption(
        "--run-gui-tests",
        action="store_true",
        default=False,
        help="Run Qt/GUI tests that are deselected by default in headless environments.",
    )


def _is_heavy_test_item(item) -> bool:
    """Heuristic for tests that can trigger OOM/host instability.

    Criteria:
        1) Known heavy file paths.
        2) Explicit ml/slow markers.
        3) Explicit timeout markers (>= 30 s).
        4) e2e marker.
    """
    # Pytest 8/9 compatibility: some collectors expose `path`, others `fspath`.
    # Fall back to nodeid to keep heavy-test isolation deterministic.
    _path_obj = getattr(item, "path", None)
    if _path_obj is not None:
        path = str(_path_obj).replace("\\", "/").lower()
    else:
        path = str(getattr(item, "fspath", "")).replace("\\", "/").lower()
    if not path:
        path = str(getattr(item, "nodeid", "")).replace("\\", "/").lower()
    if any(hint in path for hint in _HEAVY_TEST_PATH_HINTS):
        return True

    if item.get_closest_marker("ml") is not None or item.get_closest_marker("slow") is not None:
        return True

    timeout_marker = item.get_closest_marker("timeout")
    if timeout_marker and timeout_marker.args:
        try:
            timeout_s = float(timeout_marker.args[0])
            if timeout_s >= 300.0:
                return True
        except (TypeError, ValueError):
            logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

    return item.get_closest_marker("e2e") is not None


def pytest_collection_modifyitems(config, items) -> None:
    """Classify heavy tests and skip them unless explicitly enabled.

    This prevents hard machine crashes in default/local selective test runs.
    """
    run_heavy = bool(config.getoption("run_heavy_tests"))
    # Publish heavy-test mode for runtime safety guards inside processing phases.
    os.environ["AURIK_RUN_HEAVY_TESTS"] = "1" if run_heavy else "0"
    run_gui = bool(config.getoption("run_gui_tests"))
    deselected: list = []
    kept: list = []
    vscode_run = _is_vscode_run()

    for item in items:
        # GUI tests are opt-in; default runs stay deterministic in headless CI.
        if item.get_closest_marker("gui") is not None and not run_gui:
            deselected.append(item)
            continue

        if not _is_heavy_test_item(item):
            kept.append(item)
            continue

        # Ensure heavy tests are visibly classified for marker-based selection.
        item.add_marker(pytest.mark.ml)
        item.add_marker(pytest.mark.slow)

        if run_heavy:
            kept.append(item)
        else:
            deselected.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept

    if vscode_run and len(items) > _VSCODE_SAFE_TEST_LIMIT:
        vscode_kept = list(items[:_VSCODE_SAFE_TEST_LIMIT])
        vscode_deselected = list(items[_VSCODE_SAFE_TEST_LIMIT:])
        config.hook.pytest_deselected(items=vscode_deselected)
        items[:] = vscode_kept


def pytest_ignore_collect(collection_path, config):
    """Verhindert Collection/Import schwerer Test-Module und begrenzt Dateianzahl im VS Code Test Explorer.

    Dieser Hook läuft VOR dem Import der Testdatei — verhindert daher sowohl die
    Import-Kaskade (ONNX-Sessions, ~37–250 MB/Plugin) als auch den JSON-RPC-Event-Flood
    über TEST_RUN_PIPE. Beides kann VS Code bei 13 000+ Tests zum Absturz bringen.

    Datei-Limit gilt NUR wenn TEST_RUN_PIPE gesetzt ist (VS Code Test Explorer).
    Terminal-Tasks (tasks.json) haben VSCODE_PID aber kein TEST_RUN_PIPE → unbeschränkt.
    """
    global _vscode_file_count

    # Datei-Limit für VS Code Test Explorer: verhindert Import-Kaskade + Pipe-Flood.
    # Läuft BEVOR das Modul importiert wird → kein OOM, kein Event-Flood.
    if "TEST_RUN_PIPE" in os.environ:
        path_str = str(collection_path)
        if path_str.endswith(".py") and os.path.basename(path_str).startswith("test_"):
            _vscode_file_count += 1
            if _vscode_file_count > _VSCODE_FILE_LIMIT:
                return True  # Datei komplett überspringen (kein Import, kein Event)

    if bool(config.getoption("run_heavy_tests")):
        return False

    path = str(collection_path).replace("\\", "/").lower()
    basename = path.split("/")[-1]

    # Legacy-/Script-Style-Tests dürfen nie Teil der Standard-Collection sein.
    if basename in _LEGACY_IGNORE_BASENAMES:
        return True

    return any(hint in path for hint in _HEAVY_TEST_PATH_HINTS)
