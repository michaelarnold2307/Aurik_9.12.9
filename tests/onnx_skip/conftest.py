"""Skip all ONNX tests in CI where onnxruntime is unavailable.

Die Tests in tests/onnx_skip/ benötigen ONNX Runtime. In der CI-Minimal-
Umgebung (cross-platform: nur numpy/scipy/soundfile/pytest) wird onnxruntime
nicht installiert → diese Dateien werden von der Collection ausgenommen.
Lokal (onnxruntime vorhanden) laufen sie normal.

§Fix 2026-09-08: `collect_ignore = ["test_*.py"]` war wirkungslos —
collect_ignore akzeptiert konkrete Pfade, keine Glob-Muster. Ergebnis:
Collection-Fehler (ModuleNotFoundError) auf allen Cross-Platform-Jobs.
"""

import importlib.util

if importlib.util.find_spec("onnxruntime") is None:
    collect_ignore = [
        "test_onnx_advanced.py",
        "test_onnx_runtime.py",
        "test_plugin_manager.py",
    ]
