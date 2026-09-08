#!/usr/bin/env python3
"""
AURIK Professional - Main Application Entry Point
Launch the desktop application for audio restoration
"""

import faulthandler
import logging
import os
import signal
import sys
import threading
import time
import warnings
import weakref
from logging.handlers import RotatingFileHandler as _RFH
from pathlib import Path

logger = logging.getLogger(__name__)

# ── OpenBLAS/OMP thread-safety: must be set BEFORE any numpy/scipy import ────
# When multiple Python threads call into numpy simultaneously (pre-analysis
# ThreadPoolExecutor, pipeline workers), OpenBLAS spawning its own threads
# causes race conditions and segfaults (faulthandler confirmed: numpy
# _wrapreduction thread + psychoacoustic_masking_model seg-fault pattern).
# Capping OpenBLAS/OMP to 1 internal thread makes each numpy call single-
# threaded but prevents the OpenBLAS-vs-Python-thread race. PyTorch uses its
# own thread pool (set via torch.set_num_threads) and is unaffected.
# Direkte Zuweisung (nicht setdefault): überschreibt Runtime-Hook-Werte
# (runtime_hook_threading.py setzt cpu_count() via setdefault; das muss hier
# auf 1 zurückgesetzt werden, sonst OpenBLAS-Race → Heap-Corruption → SIGABRT)
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
# Release-Start: Framework-Warnungen ohne Nutzerwert ausblenden (§release_invariants).
os.environ.setdefault("MIOPEN_LOG_LEVEL", "1")

# Release-Start: bekannte Framework-Hinweise ohne Nutzerwert ausblenden.
warnings.filterwarnings(
    "ignore",
    message=r"Importing from timm\.models\.layers is deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"torch\.meshgrid: in an upcoming release.*",
    category=UserWarning,
)

# ── NumPy: Mean of empty slice + divide-by-zero bei Stille/Silence-Passagen ───
# Diese RuntimeWarnings sind harmlos (betreffen nur leere Arrays in Phasen mit
# 0-Energy-Segmenten). Sie werden korrekt per np.clip / np.nan_to_num behandelt.
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide")

# ── HuggingFace/MERT: Harmlose Weight-Mismatch-Warnung beim Laden von Checkpoints ──
warnings.filterwarnings("ignore", message="Some weights of the model checkpoint.*were not used")
warnings.filterwarnings("ignore", message="Some weights of.*were not initialized from the model checkpoint")
warnings.filterwarnings("ignore", message="You should probably TRAIN this model")

# ── webrtcvad: pkg_resources ist deprecated (externes Package, nicht Aurik) ──
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated.*", category=UserWarning)

# ── torchaudio/torch: deprecation warnings aus externen Packages ──
warnings.filterwarnings("ignore", message=".*onesided.*deprecated.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*", category=FutureWarning)

# ── PyTorch global thread-pool — must be set once at startup (§VERBOTEN) ─────
try:
    import torch as _torch

    _torch.set_num_threads(os.cpu_count() or 4)
except Exception:  # torch not installed / import error on first run
    logger.debug("Stiller optionaler Ausnahmefall ignoriert", exc_info=True)

# Add parent + plugins directory to path (für PANNs, VERSA, Bridge-Plugins)
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "plugins"))

# ── Logging setup: file + console ─────────────────────────────────────────────
_LOG_DIR = Path(__file__).parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_log_file = _LOG_DIR / "aurik_backend.log"

_root_logger = logging.getLogger()
_root_logger.setLevel(logging.DEBUG)

# AURIK_DEBUG=1 → all handlers at DEBUG, dedicated timestamped log, full source info
_DEBUG_MODE = os.getenv("AURIK_DEBUG", "0") == "1"

# File handler (5 MB, rotating)
_fh = _RFH(str(_log_file), maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
_fh.setLevel(logging.DEBUG if _DEBUG_MODE else logging.INFO)
_fh.setFormatter(
    logging.Formatter(
        "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s (%(filename)s:%(lineno)d): %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if _DEBUG_MODE
    else logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
)
_root_logger.addHandler(_fh)

# Debug session log: separate file with timestamp — never rotated, always complete
if _DEBUG_MODE:
    import datetime as _dt

    _debug_log_file = _LOG_DIR / f"aurik_debug_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _dfh = logging.FileHandler(str(_debug_log_file), mode="w", encoding="utf-8")
    _dfh.setLevel(logging.DEBUG)
    _dfh.setFormatter(
        logging.Formatter(
            "[%(asctime)s.%(msecs)03d] %(levelname)-8s %(name)s (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    _root_logger.addHandler(_dfh)
    # Path for shell-side tail -f: store path in well-known file
    (_LOG_DIR / "aurik_debug_latest.log").write_text(str(_debug_log_file), encoding="utf-8")

    # ── Noisy 3rd-party loggers: cap at WARNING — no DSP/Aurik diagnostic value ─
    # Numba JIT compiler emits hundreds of DEBUG lines on first compilation.
    # PIL/fonttools/matplotlib occasionally imported by Qt resources.
    # triton/torch.fx are pure framework internals with no Aurik diagnostic value.
    for _noisy_logger in (
        "numba",
        "numba.core",
        "numba.core.byteflow",
        "numba.core.interpreter",
        "numba.core.ssa",
        "numba.core.analysis",
        "numba.typed",
        "PIL",
        "fonttools",
        "matplotlib",
        "matplotlib.font_manager",
        "triton",
        "torch.fx",
        "torch._dynamo",
        "urllib3",
        "h5py",
        "audioread",
        "librosa",
        "librosa.core",
        "pedalboard",
        "backend.core.erb_auditory_masking",
        "dsp.pghi",
    ):
        logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# Console handler (INFO+ für Live-Phasen-Status in Normalmodus)
_ch = logging.StreamHandler(sys.stderr)
_ch.setLevel(logging.INFO)
_ch.setFormatter(
    logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
)
_root_logger.addHandler(_ch)
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)

# Must outlive _enable_crash_forensics() — faulthandler holds a raw FILE* not a Python
# The faulthandler file handle must be kept alive until the process exits.
# Python docs: "The file must be kept open until the fault handler is disabled."
_crash_log_fh: list = [None]  # mutable container avoids a global statement


def _enable_crash_forensics() -> None:
    """Aktiviert native-crash diagnostics for hard faults (SIGSEGV/SIGABRT)."""
    try:
        crash_log = _LOG_DIR / "python_faulthandler.log"
        _crash_log_fh[0] = open(crash_log, "a", encoding="utf-8")
        faulthandler.enable(file=_crash_log_fh[0], all_threads=True)
        for _sig in (signal.SIGSEGV, signal.SIGABRT, signal.SIGBUS, signal.SIGFPE):
            try:
                faulthandler.register(_sig, file=_crash_log_fh[0], all_threads=True)
            except Exception:
                # Not all platforms allow explicit registration for every signal.
                logger.debug("main: faulthandler signal registration skipped (platform limitation)", exc_info=True)
        logger.info("Crash forensics active (faulthandler): %s", crash_log)
    except Exception as exc:
        logger.warning("Faulthandler setup fehlgeschlagen (non-fatal): %s", exc)

    # §v10.993: Globalen Exception-Hook installieren — unbehandelte GUI-Fehler
    # landen sonst nur unsichtbar in stderr. Ohne diesen Hook ist der
    # Crash-Reporter toter Code für die GUI.
    try:
        from backend.api.bridge import install_crash_handler

        install_crash_handler()
    except Exception as exc:
        logger.warning("Crash-Reporter-Installation fehlgeschlagen (non-fatal): %s", exc)


# PyQt5 and Aurik imports must come after logging setup and sys.path configuration.
# pylint: disable=wrong-import-position
# pylint: disable=no-name-in-module
from PyQt5.QtCore import Qt, QTimer, qInstallMessageHandler  # type: ignore[attr-defined]
from PyQt5.QtGui import QIcon  # type: ignore[attr-defined]
from PyQt5.QtWidgets import QApplication, QMessageBox  # type: ignore[attr-defined]

# pylint: enable=no-name-in-module
from Aurik10 import __version__
from Aurik10.i18n import t
from Aurik10.ui.modern_window import ModernMainWindow

# pylint: enable=wrong-import-position

_main_window_ref: weakref.ReferenceType[ModernMainWindow] | None = None


def _run_startup_model_check(_app: QApplication) -> None:
    """Prüft ML models before window creation — shows dialog on problems (non-fatal).

    Non-blocking for missing optional models.
    Warning for missing primary models (DSP fallback active).
    """
    try:
        from backend.api.bridge import (  # pylint: disable=import-outside-toplevel
            get_model_downloader,
            get_startup_check_result,
        )

        def run_self_heal(missing_primary, missing_optional):
            dl = get_model_downloader()
            to_repair = [m["name"] for m in (missing_primary + missing_optional)]
            for name in to_repair:
                entry = dl.get_entry(name)
                if entry is not None:
                    # Download SOTA-Upgrade if configured, otherwise use bundled
                    dl.schedule_sota_upgrade(entry)
                # Yield to Qt event loop between iterations
                _app.processEvents()
                time.sleep(0.2)
            # Wait for all downloads to complete; split into 10 × 200 ms slices
            # so Qt events continue to be processed (no UI freeze).
            for _ in range(10):
                _app.processEvents()
                time.sleep(0.2)

        result = get_startup_check_result()
        if result is None:
            return
        if not result.all_ok and result.user_message_de:
            icon = QMessageBox.Icon.Critical if result.is_critical else QMessageBox.Icon.Warning
            box = QMessageBox()
            box.setWindowTitle("AURIK — " + result.user_title_de)
            box.setText(result.user_message_de)
            box.setIcon(icon)
            # Additional self-healing button
            repair_btn = box.addButton("Modelle reparieren", QMessageBox.ActionRole)
            box.setStandardButtons(QMessageBox.StandardButton.Ok)
            box.setDefaultButton(QMessageBox.StandardButton.Ok)
            box.exec_()
            if box.clickedButton() == repair_btn:
                # Start automatic self-healing
                run_self_heal(result.missing_primary, result.missing_optional)
                # Re-run check after completion
                result2 = get_startup_check_result()
                if result2 is not None and result2.all_ok:
                    QMessageBox.information(
                        None,
                        "AURIK — Modelle repariert",
                        "Alle ML-Modelle wurden erfolgreich repariert und geladen."
                        " Die Restaurierung ist jetzt freigeschaltet.",
                    )
                else:
                    QMessageBox.critical(
                        None,
                        "AURIK — Reparatur fehlgeschlagen",
                        "Einige Modelle konnten nicht repariert werden."
                        " Bitte prüfen Sie Ihre Internetverbindung"
                        " oder wenden Sie sich an den Support.",
                    )
    except Exception as exc:
        # Startup check must never block app startup
        logger.warning("Start model Pruefung fehlgeschlagen (non-fatal): %s", exc)


def _warmup_models_background() -> None:
    """Start background warmup of ML models — non-blocking daemon thread."""

    def _run() -> None:
        try:
            from backend.api.bridge import (
                warmup_models_background as _wb,  # type: ignore[import]  # pylint: disable=import-outside-toplevel
            )

            _wb()
        except Exception as _e:
            logger.debug("Warmup fehlgeschlagen (non-fatal): %s", _e)

    t = threading.Thread(target=_run, daemon=True, name="AurikWarmup")
    t.start()


# ---------------------------------------------------------------------------
# §3.9.2  SIGTERM handler — checkpoint + graceful Qt shutdown
# ---------------------------------------------------------------------------


def _emergency_checkpoint_if_running() -> None:
    """Non-blocking best-effort checkpoint on SIGTERM (§3.9.2).

    Only fires when a BatchProcessingThread is currently running.
    Uses threading.Event.wait(timeout=0) to avoid blocking the signal handler.
    Writes checkpoint atomically (.tmp → os.replace) if audio is available.
    """
    try:
        window = _main_window_ref() if _main_window_ref is not None else None
        if not isinstance(window, ModernMainWindow):
            return
        bt = getattr(window, "batch_thread", None)
        if bt is not None and bt.isRunning():
            try:
                # best-effort: trigger checkpoint without waiting
                save_fn = getattr(bt, "request_emergency_checkpoint", None)
                if callable(save_fn):
                    save_fn()
            except Exception as _ce:
                logger.debug("Emergency checkpoint fehlgeschlagen (non-fatal): %s", _ce)
    except Exception as _exc:
        logger.debug("_emergency_checkpoint_if_laeuft uebersprungen: %s", _exc)


_sigterm_shutdown_started = threading.Event()


def _sigterm_handler(signum: int, _frame: object) -> None:
    """SIGTERM → emergency checkpoint + graceful Qt shutdown (§3.9.2).

    CRITICAL: Signal handlers are re-entrant-unsafe. This handler ONLY schedules
    work via QTimer.singleShot (or calls fallback synchronously). Minimal work
    in the handler itself. The _sigterm_shutdown_started flag prevents
    re-entry of the finalization logic.
    """
    if _sigterm_shutdown_started.is_set():
        # SIGTERM came in again while shutdown was in progress.
        # Just return; don't hold the signal handler or do heavy work.
        return
    # Set immediately to prevent re-entry from concurrent SIGTERMs
    _sigterm_shutdown_started.set()
    logger.warning("SIGTERM received (signum=%d) — initiating emergency Herunterfahren", signum)

    def _finalize_sigterm_shutdown() -> None:
        """Finalize shutdown: checkpoint → quit."""
        _emergency_checkpoint_if_running()
        try:
            _app2 = QApplication.instance()
            if _app2:
                _app2.quit()
        except Exception as _exc:
            logger.debug("Qt-Quit after SIGTERM fehlgeschlagen: %s", _exc)
        # If Qt.quit fails, process will exit from SIGTERM anyhow

    try:
        _app = QApplication.instance()
        if _app:
            # Schedule via event loop (non-blocking from signal handler)
            QTimer.singleShot(0, _finalize_sigterm_shutdown)
            return
    except Exception as _exc:
        logger.debug("SIGTERM QTimer dispatch fehlgeschlagen, Ersatzpfad: %s", _exc)

    # Fallback if QTimer dispatch failed (e.g., no event loop)
    _finalize_sigterm_shutdown()


def _process_events_ms(app: "QApplication", ms: int) -> None:
    """Verarbeitet Qt events for *ms* milliseconds — keeps splash animation alive."""
    deadline = time.monotonic() + ms / 1000.0
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.016)  # ~60 fps


def main():
    """Launch AURIK Professional"""
    global _main_window_ref

    _enable_crash_forensics()

    # Linux hardening: prefer software OpenGL to reduce GPU driver related Qt crashes.
    # Can be disabled for troubleshooting via: AURIK_FORCE_SOFTWARE_OPENGL=0
    _force_sw_gl = os.getenv("AURIK_FORCE_SOFTWARE_OPENGL", "1") == "1"

    # Enable high DPI scaling (PyQt5 stubs do not expose these attributes -> ignore)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)  # type: ignore[attr-defined]
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)  # type: ignore[attr-defined]
    if sys.platform.startswith("linux") and _force_sw_gl:
        QApplication.setAttribute(Qt.AA_UseSoftwareOpenGL, True)  # type: ignore[attr-defined]
    # Prevent XDG/GTK portal hang on Linux: native file dialogs must not be used
    # (portal daemon can block the main thread before DontUseNativeDialog takes effect)
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, True)  # type: ignore[attr-defined]

    app = QApplication(sys.argv)
    app.setApplicationName("AURIK Professional")
    app.setOrganizationName("AURIK")
    app.setApplicationVersion(__version__)

    # §3.9.2: Register SIGTERM handler for graceful shutdown + emergency checkpoint.
    # Must be installed after QApplication to avoid race with Qt's signal handling.
    signal.signal(signal.SIGTERM, _sigterm_handler)

    # Application icon (taskbar, dock, alt-tab)
    _res = Path(__file__).parent / "resources"
    for _icon_path in (_res / "vinyl_gold.png", _res / "icon_premium.svg", _res / "icon.png"):
        if _icon_path.exists():
            app.setWindowIcon(QIcon(str(_icon_path)))
            break

    # Set dark theme style
    app.setStyle("Fusion")

    # Capture Qt warnings/errors into Python logs for post-mortem analysis.
    try:

        def _qt_msg_handler(mode, context, message):
            _file = getattr(context, "file", "?") if context is not None else "?"
            _line = getattr(context, "line", 0) if context is not None else 0
            logger.warning("QtMessage[%s] %s:%s %s", int(mode), _file, _line, message)

        qInstallMessageHandler(_qt_msg_handler)
    except Exception as exc:
        logger.debug("Qt message handler setup uebersprungen: %s", exc)

    # Global QToolTip styling — prevents the default black/system-colored tooltip box.
    # Border uses the app's purple accent; background matches the dark UI palette.
    app.setStyleSheet(
        "QToolTip {"
        "  background-color: #1a1a2e;"
        "  color: #d8d8f0;"
        "  border: 1px solid #4a3878;"
        "  border-radius: 6px;"
        "  padding: 6px 10px;"
        "  font-size: 9pt;"
        "}"
    )

    # ── Splash screen ─────────────────────────────────────────────────────────
    splash = None
    try:
        from Aurik10.ui.splash_screen import AurikSplashScreen  # pylint: disable=import-outside-toplevel

        splash = AurikSplashScreen()
        splash.setWindowOpacity(0.0)
        splash.show()
        splash.raise_()  # bring to front without WindowStaysOnTopHint
        app.processEvents()

        # Fade in: 32 steps × 18 ms ≈ 580 ms
        for _step in range(32):
            splash.setWindowOpacity((_step + 1) / 32.0)
            app.processEvents()
            time.sleep(0.018)

        _process_events_ms(app, 400)  # hold visible for a moment (was 120 ms)

    except Exception as _exc:
        # Splash must never block the application from starting
        logger.warning("Splash screen could not be geladen (non-fatal): %s", _exc)
        splash = None

    # ── GPU-Erkennung im Hauptthread (§v10.304.30) ─────────────────────────
    # Muss VOR ModernMainWindow und VOR allen Worker-Threads laufen.
    # torch.cuda.init() muss im Hauptthread erfolgen — sonst hängt
    # ROCm/HIP bei der ersten GPU-Operation im Worker-Thread.
    # AURIK_FORCE_CPU=1 überspringt die Erkennung komplett.
    if splash:
        splash.set_status(t("splash.status.gpu"))
        app.processEvents()
    try:
        from backend.api.bridge import get_ml_device_manager as _gpu_mgr

        _mgr = _gpu_mgr()
        _mgr.wait_for_detection(timeout=5.0)  # type: ignore[call-arg]
        logger.info("main: GPU-Erkennung abgeschlossen — %s", _mgr.gpu_name)
    except Exception as _gpu_exc:
        logger.debug("main: GPU-Erkennung fehlgeschlagen (CPU-only): %s", _gpu_exc)

    # ── Startup model check ───────────────────────────────────────────────────
    if splash:
        splash.set_status(t("splash.status.models"))
        app.processEvents()

    _run_startup_model_check(app)

    # ── librosa numba JIT warmup (main thread) ────────────────────────────────
    # numba 0.64.0 gufunc JIT in librosa is thread-unsafe on first call.
    # Fix: warm up numba JIT on the main thread using librosa.feature (avoids
    # the get_call_template bug in numba's dispatcher for gufunc paths).
    try:
        from io import BytesIO as _BytesIO

        import librosa as _librosa_warmup
        import numpy as _np_warmup
        import soundfile as _sf_warmup

        # Use melspectrogram warmup (avoids numba gufunc dispatcher bug)
        _dummy = _np_warmup.zeros(2048, dtype=_np_warmup.float32)
        _dummy[512:1536] = _np_warmup.sin(2 * _np_warmup.pi * 440 * _np_warmup.arange(1024) / 22050) * 0.1
        _librosa_warmup.feature.melspectrogram(y=_dummy, sr=22050, n_mels=128)
        # Also warm up resample via the non-gufunc path
        try:
            _librosa_warmup.resample(_dummy[:512], orig_sr=22050, target_sr=48000)
        except AttributeError:
            pass  # numba dispatcher issue on this path — already handled above

        logger.info("librosa numba JIT warmup vollstaendig (main thread)")
        del _dummy, _np_warmup, _librosa_warmup, _sf_warmup, _BytesIO
    except Exception as _wup_exc:
        logger.debug("librosa warmup uebersprungen (non-fatal): %s", _wup_exc)

    # ── Build main window ─────────────────────────────────────────────────────
    if splash:
        splash.set_status(t("splash.status.ui"))
        app.processEvents()

    window = ModernMainWindow()
    _main_window_ref = weakref.ref(window)

    if splash:
        splash.set_status(t("splash.status.ready"))
        app.processEvents()
        _process_events_ms(app, 600)  # show "Bereit." briefly (was 280 ms)

    # Show main window beneath splash, then fade out splash
    window.show()
    app.processEvents()

    if splash:
        # Fade out: 28 steps × 20 ms ≈ 560 ms
        # The main window becomes visible as splash becomes transparent.
        splash.stop_animation()
        for _step in range(28):
            splash.setWindowOpacity(1.0 - (_step + 1) / 28.0)
            app.processEvents()
            time.sleep(0.020)
        splash.close()
        splash = None

    # §9.7.4 Model warmup: ModernMainWindow.__init__ starts via QTimer.singleShot(2000)
    # warmup_models_background() from backend.api.bridge — no second thread here.
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
